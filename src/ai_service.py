import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from google import genai
from google.genai import types

from src.rag import (
    DEFAULT_GENERATION_MODEL,
    SongDocument,
    keyword_retrieve,
    load_cache,
    retrieve,
)
from src.recommender import recommend_songs
from src.tools import search_itunes

MAX_TOOL_CALLS = 3
DEFAULT_CACHE_PATH = ".cache/song_embeddings.json"

_TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="search_catalog",
        description=(
            "Keyword-search the local song catalog for tracks matching a vibe, "
            "mood, genre, or listening context. Use when the initial evidence "
            "looks incomplete."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms"},
                "limit": {"type": "integer", "description": "Max results (1-10)"},
            },
            "required": ["query"],
        },
    ),
    types.FunctionDeclaration(
        name="search_itunes",
        description=(
            "Live-search the iTunes catalog for real-world tracks beyond the "
            "local catalog. Use for discovery requests or genres the local "
            "catalog lacks."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms"},
                "limit": {"type": "integer", "description": "Max results (1-10)"},
            },
            "required": ["query"],
        },
    ),
    types.FunctionDeclaration(
        name="get_song_details",
        description="Get the full local catalog record for one song by its song_id (e.g. local-3).",
        parameters={
            "type": "object",
            "properties": {
                "song_id": {"type": "string", "description": "Song id like local-3"},
            },
            "required": ["song_id"],
        },
    ),
]


@dataclass
class RecommendationResult:
    recommendations: List[Dict[str, Any]]
    summary: str
    fallback_used: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


def build_system_prompt() -> str:
    return (
        "You are VibeMatch, a music recommendation assistant. "
        "You recommend songs ONLY from the provided evidence — local catalog results "
        "and live iTunes search results. Never invent songs. "
        "For every recommendation you MUST cite the source (local or itunes) and "
        "the exact evidence (genre match, mood match, retrieval snippet, etc.). "
        "You may call the provided tools to search the local catalog or the live "
        "iTunes catalog when the initial evidence is insufficient. "
        "Keep responses concise. When you are done using tools, return valid JSON "
        "with keys: recommendations (array), summary (string). "
        "Each recommendation object must have: title, artist, reason, source, evidence."
    )


def build_grounding_prompt(
    user_query: str,
    local_docs: Sequence[SongDocument],
    itunes_results: Sequence[Dict[str, Any]],
    user_prefs: Optional[Dict[str, Any]] = None,
) -> str:
    lines = [
        "USER REQUEST: " + user_query,
        "",
        "USER PREFERENCES: " + json.dumps(user_prefs or {}),
        "",
        "LOCAL CATALOG EVIDENCE:",
    ]
    for doc in local_docs:
        lines.append(
            f"- [{doc.song_id}] {doc.title} by {doc.artist} | genre={doc.genre} mood={doc.mood} energy={doc.energy} | {doc.description}"
        )
    lines.append("")
    lines.append("LIVE ITUNES EVIDENCE:")
    for item in itunes_results:
        lines.append(
            f"- {item.get('title', '')} by {item.get('artist', '')} | genre={item.get('genre', '')} | source=itunes"
        )
    lines.append("")
    lines.append(
        "Recommend up to 5 songs from the evidence above. "
        "Prefer local catalog when it fits; call tools if you need more evidence. "
        "Cite source and evidence for each. Return ONLY valid JSON when finished."
    )
    return "\n".join(lines)


class VibeMatchService:
    def __init__(
        self,
        docs: Sequence[SongDocument],
        client: Optional[genai.Client] = None,
        cache_path: Optional[str] = DEFAULT_CACHE_PATH,
    ):
        self.docs = list(docs)
        self.client = client or genai.Client()
        self.cache_embeddings: Dict[str, Sequence[float]] = {}
        if cache_path and Path(cache_path).exists():
            try:
                cache = load_cache(Path(cache_path))
                self.cache_embeddings = cache.get("embeddings", {})
            except (json.JSONDecodeError, OSError, KeyError):
                self.cache_embeddings = {}

    def recommend(
        self,
        query: str,
        user_prefs: Optional[Dict[str, Any]] = None,
        k: int = 5,
        mode: str = "balanced",
        include_live: bool = False,
        offline: bool = False,
    ) -> RecommendationResult:
        user_prefs = user_prefs or {}

        # 1. Retrieve local evidence
        retrieval = retrieve(
            query,
            self.docs,
            cache_embeddings=self.cache_embeddings or None,
            k=k,
            filters=user_prefs,
            use_llm_expansion=False,
            offline=offline,
        )
        local_docs = retrieval.documents

        # 2. Optionally fetch live iTunes discoveries up front
        itunes_results: List[Dict[str, Any]] = []
        if include_live:
            itunes_results = search_itunes(query, limit=k)

        # 3. Bounded tool-calling loop, then grounded answer
        prompt = build_grounding_prompt(query, local_docs, itunes_results, user_prefs)
        try:
            data, tool_calls, tool_itunes = self._run_tool_loop(prompt, k)
            itunes_results = itunes_results + tool_itunes
            if data is None:
                return self._fallback(
                    query, user_prefs, k, mode, "tool loop exhausted",
                    tool_calls=tool_calls, tool_loop_exhausted=True,
                )

            recommendations = self._filter_hallucinations(
                data.get("recommendations", []),
                itunes_results,
            )
            recommendations = self._enrich_itunes_recs(recommendations, itunes_results)
            summary = data.get("summary", "")
            local_count = sum(1 for r in recommendations if r.get("source") == "local")
            itunes_count = sum(1 for r in recommendations if r.get("source") == "itunes")
            return RecommendationResult(
                recommendations=recommendations[:k],
                summary=summary,
                fallback_used=False,
                metadata={
                    "local_count": local_count,
                    "itunes_count": itunes_count,
                    "tool_calls": tool_calls,
                    "debug": retrieval.debug,
                },
            )
        except Exception as exc:
            return self._fallback(query, user_prefs, k, mode, str(exc))

    def _run_tool_loop(
        self, prompt: str, k: int
    ) -> tuple[Optional[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
        """Bounded multi-step tool loop.

        Returns (parsed_json, tool_call_names, itunes_items_from_tools).
        (None, calls, items) means the model never stopped calling tools.
        """
        contents: List[Any] = [prompt]
        tool_calls: List[str] = []
        tool_itunes: List[Dict[str, Any]] = []
        config = types.GenerateContentConfig(
            system_instruction=build_system_prompt(),
            temperature=0.2,
            tools=[types.Tool(function_declarations=_TOOL_DECLARATIONS)],
        )

        response = None
        for _ in range(MAX_TOOL_CALLS):
            response = self.client.models.generate_content(
                model=DEFAULT_GENERATION_MODEL,
                contents=contents,
                config=config,
            )
            calls = getattr(response, "function_calls", None)
            if not isinstance(calls, list) or not calls:
                break
            parts = []
            for call in calls:
                tool_calls.append(call.name)
                output = self._dispatch_tool(call.name, dict(call.args or {}), k)
                if call.name == "search_itunes" and isinstance(output, list):
                    tool_itunes.extend(output)
                parts.append(
                    types.Part.from_function_response(
                        name=call.name, response={"result": output}
                    )
                )
            if response.candidates:
                contents.append(response.candidates[0].content)
            contents.append(types.Content(role="tool", parts=parts))
        else:
            return None, tool_calls, tool_itunes

        raw = (response.text or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw), tool_calls, tool_itunes

    def _dispatch_tool(self, name: str, args: Dict[str, Any], k: int) -> Any:
        limit = max(1, min(int(args.get("limit", k)), 10))
        if name == "search_itunes":
            return search_itunes(args.get("query", ""), limit=limit)
        if name == "search_catalog":
            results = keyword_retrieve(args.get("query", ""), self.docs, k=limit)
            return [
                {
                    "song_id": doc.song_id,
                    "title": doc.title,
                    "artist": doc.artist,
                    "genre": doc.genre,
                    "mood": doc.mood,
                    "energy": doc.energy,
                    "description": doc.description,
                }
                for doc in results
            ]
        if name == "get_song_details":
            song_id = args.get("song_id", "")
            for doc in self.docs:
                if doc.song_id == song_id:
                    return {
                        "song_id": doc.song_id,
                        "title": doc.title,
                        "artist": doc.artist,
                        "genre": doc.genre,
                        "mood": doc.mood,
                        "energy": doc.energy,
                        "description": doc.description,
                        "listening_context": doc.listening_context,
                    }
            return {"error": f"unknown song_id {song_id}"}
        return {"error": f"unknown tool {name}"}

    def _filter_hallucinations(
        self,
        recommendations: List[Dict[str, Any]],
        itunes_results: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        # Guard against invented songs: anything not in the full catalog or
        # the live iTunes result set is dropped, even if the model complied
        # with an injection attempt.
        known_titles = {doc.title.lower() for doc in self.docs}
        known_titles.update(item.get("title", "").lower() for item in itunes_results)
        return [
            rec
            for rec in recommendations
            if rec.get("title", "").lower() in known_titles
        ]

    def _enrich_itunes_recs(
        self,
        recommendations: List[Dict[str, Any]],
        itunes_results: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        by_title = {item.get("title", "").lower(): item for item in itunes_results}
        enriched = []
        for rec in recommendations:
            item = by_title.get(rec.get("title", "").lower())
            if item:
                for key in ("artwork_url", "preview_url", "track_view_url", "album"):
                    if item.get(key):
                        rec[key] = item[key]
            enriched.append(rec)
        return enriched

    def _fallback(
        self,
        query: str,
        user_prefs: Dict[str, Any],
        k: int,
        mode: str,
        error: str,
        tool_calls: Optional[List[str]] = None,
        tool_loop_exhausted: bool = False,
    ) -> RecommendationResult:
        songs = [doc.raw for doc in self.docs if doc.raw]
        ranked = recommend_songs(user_prefs, songs, k=k, mode=mode)
        recommendations = []
        for song, score, explanation in ranked:
            recommendations.append(
                {
                    "title": song["title"],
                    "artist": song["artist"],
                    "reason": f"Deterministic match (score {score:.2f})",
                    "source": "local",
                    "evidence": explanation,
                }
            )
        metadata: Dict[str, Any] = {
            "error": error,
            "local_count": len(recommendations),
            "itunes_count": 0,
            "tool_calls": tool_calls or [],
        }
        if tool_loop_exhausted:
            metadata["tool_loop_exhausted"] = True
        return RecommendationResult(
            recommendations=recommendations,
            summary=f"Gemini unavailable; showing deterministic recommendations for '{query}'.",
            fallback_used=True,
            metadata=metadata,
        )
