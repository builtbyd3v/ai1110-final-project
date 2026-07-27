import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from google import genai
from google.genai import types

from src.rag import DEFAULT_GENERATION_MODEL, SongDocument, retrieve
from src.recommender import recommend_songs
from src.tools import search_itunes

MAX_TOOL_CALLS = 3


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
        "Keep responses concise. Return valid JSON with keys: recommendations (array), summary (string). "
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
            f"- {doc.title} by {doc.artist} | genre={doc.genre} mood={doc.mood} energy={doc.energy} | {doc.description}"
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
        "Prefer local catalog when it fits; use itunes for discovery beyond the catalog. "
        "Cite source and evidence for each. Return ONLY valid JSON."
    )
    return "\n".join(lines)


class VibeMatchService:
    def __init__(
        self,
        docs: Sequence[SongDocument],
        client: Optional[genai.Client] = None,
    ):
        self.docs = list(docs)
        self.client = client or genai.Client()

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
            query, self.docs, k=k, filters=user_prefs,
            use_llm_expansion=False, offline=offline,
        )
        local_docs = retrieval.documents

        # 2. Optionally fetch live iTunes discoveries
        itunes_results = []
        if include_live:
            itunes_results = search_itunes(query, limit=k)

        # 3. Build grounded prompt and call Gemini
        prompt = build_grounding_prompt(query, local_docs, itunes_results, user_prefs)
        try:
            response = self.client.models.generate_content(
                model=DEFAULT_GENERATION_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=build_system_prompt(),
                    temperature=0.2,
                ),
            )
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)

            recommendations = self._filter_hallucinations(
                data.get("recommendations", []),
                itunes_results,
            )
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
                    "debug": retrieval.debug,
                },
            )
        except Exception as exc:
            return self._fallback(query, user_prefs, k, mode, str(exc))

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

    def _fallback(
        self,
        query: str,
        user_prefs: Dict[str, Any],
        k: int,
        mode: str,
        error: str,
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
        return RecommendationResult(
            recommendations=recommendations,
            summary=f"Gemini unavailable; showing deterministic recommendations for '{query}'.",
            fallback_used=True,
            metadata={"error": error, "local_count": len(recommendations), "itunes_count": 0},
        )
