import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

from src.recommender import load_songs, recommend_songs, song_to_document

load_dotenv()

DEFAULT_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
DEFAULT_GENERATION_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
RRF_K = 60.0

_EXPANSION_HINTS = {
    "sad": ["melancholic", "lonely", "heartbreak", "late-night reflection"],
    "happy": ["euphoric", "upbeat", "sunny", "feel-good"],
    "chill": ["calm", "cozy", "mellow", "slow"],
    "study": ["focus", "deep work", "instrumental", "lo-fi", "studying"],
    "studying": ["study", "focus", "deep work", "instrumental", "lo-fi"],
    "workout": ["gym", "energetic", "intense", "driving", "workouts"],
    "workouts": ["workout", "gym", "energetic", "intense", "driving"],
    "rain": ["rainy", "stormy", "grey afternoon", "indoor hours"],
    "rainy": ["rain", "stormy", "grey afternoon", "indoor hours"],
    "drive": ["night drive", "road trip", "highway", "synthwave"],
    "drives": ["night drive", "road trip", "highway", "synthwave"],
    "love": ["romantic", "sensual", "date night", "intimate"],
}


@dataclass
class SongDocument:
    song_id: str
    title: str
    artist: str
    source: str
    text: str
    description: str = ""
    listening_context: str = ""
    genre: str = ""
    mood: str = ""
    energy: float = 0.0
    raw: Optional[Dict[str, Any]] = None


@dataclass
class RetrievalResult:
    documents: List[SongDocument]
    debug: Dict[str, Any]


def _song_to_document(song: Dict[str, Any]) -> SongDocument:
    return SongDocument(
        song_id=f"local-{song['id']}",
        title=song["title"],
        artist=song["artist"],
        source="local",
        text=song_to_document(song),
        description=song.get("description", ""),
        listening_context=song.get("listening_context", ""),
        genre=song.get("genre", ""),
        mood=song.get("mood", ""),
        energy=song.get("energy", 0.0),
        raw=song,
    )


def build_documents(songs: Iterable[Dict[str, Any]]) -> List[SongDocument]:
    return [_song_to_document(song) for song in songs]


def load_documents_from_csv(csv_path: str) -> List[SongDocument]:
    return build_documents(load_songs(csv_path))


def _tokenize(text: str) -> List[str]:
    normalized = (
        text.lower()
        .replace(";", " ")
        .replace(",", " ")
        .replace(":", " ")
        .replace("-", " ")
    )
    return [token for token in normalized.split() if token]


def _matches_filters(doc: SongDocument, filters: Optional[Dict[str, Any]]) -> bool:
    if not filters:
        return True
    genre = filters.get("genre")
    mood = filters.get("mood")
    if genre and doc.genre.lower() != str(genre).lower():
        return False
    if mood and doc.mood.lower() != str(mood).lower():
        return False
    return True


def _expand_tokens(tokens: Sequence[str]) -> List[str]:
    expanded = set(tokens)
    for token in tokens:
        for related in _EXPANSION_HINTS.get(token, []):
            expanded.add(related)
    return sorted(expanded)


def _keyword_score(query_tokens: Sequence[str], doc_tokens: Sequence[str]) -> float:
    doc_counts: Dict[str, int] = {}
    for token in doc_tokens:
        doc_counts[token] = doc_counts.get(token, 0) + 1

    exact_overlap = sum(doc_counts.get(token, 0) for token in query_tokens)
    coverage = sum(1 for token in query_tokens if token in doc_counts)
    partial_overlap = 0.0
    for token in query_tokens:
        if token in doc_counts:
            continue
        if any(len(token) >= 4 and token in doc_token or (len(doc_token) >= 4 and doc_token in token) for doc_token in doc_counts):
            partial_overlap += 0.5
    return exact_overlap + coverage + partial_overlap


def keyword_retrieve(
    query: str,
    docs: Sequence[SongDocument],
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> List[SongDocument]:
    """BM25-style lightweight keyword retrieval using token overlap scoring."""
    base_tokens = _tokenize(query)
    query_tokens = _expand_tokens(base_tokens)
    if not query_tokens:
        return []

    scored = []
    for doc in docs:
        if not _matches_filters(doc, filters):
            continue
        doc_tokens = _tokenize(doc.text)
        if not doc_tokens:
            continue
        score = _keyword_score(query_tokens, doc_tokens)
        if score <= 0:
            continue
        scored.append((score, doc))
    scored.sort(key=lambda item: (-item[0], item[1].song_id))
    if len(scored) < k:
        seen = {doc.song_id for _, doc in scored}
        for doc in docs:
            if not _matches_filters(doc, filters) or doc.song_id in seen:
                continue
            scored.append((0.0, doc))
            if len(scored) >= k:
                break
    return [doc for _, doc in scored[:k]]


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def vector_retrieve(
    query_embedding: Sequence[float],
    docs: Sequence[SongDocument],
    embeddings: Dict[str, Sequence[float]],
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> List[SongDocument]:
    scored = []
    for doc in docs:
        if not _matches_filters(doc, filters):
            continue
        doc_embedding = embeddings.get(doc.song_id)
        if doc_embedding is None:
            continue
        score = _cosine_similarity(query_embedding, doc_embedding)
        scored.append((score, doc))
    scored.sort(key=lambda item: (-item[0], item[1].song_id))
    return [doc for _, doc in scored[:k]]


def normalize_ranking(docs: Sequence[SongDocument]) -> Dict[str, float]:
    if not docs:
        return {}
    if len(docs) == 1:
        return {docs[0].song_id: 1.0}
    return {
        doc.song_id: 1.0 - (index / (len(docs) - 1))
        for index, doc in enumerate(docs)
    }


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SongDocument]],
    k: int = 5,
) -> List[SongDocument]:
    fused: Dict[str, Tuple[float, SongDocument]] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            current_score, existing_doc = fused.get(doc.song_id, (0.0, doc))
            fused[doc.song_id] = (current_score + 1.0 / (RRF_K + rank), existing_doc)
    ordered = sorted(fused.values(), key=lambda item: (-item[0], item[1].song_id))
    return [doc for _, doc in ordered[:k]]


def fuse_rankings(
    rankings: Sequence[Sequence[SongDocument]],
    weights: Optional[Sequence[float]] = None,
    k: int = 5,
) -> List[SongDocument]:
    if not rankings:
        return []
    weights = list(weights or [1.0] * len(rankings))
    if len(weights) != len(rankings):
        raise ValueError("weights must match rankings length")
    weighted_scores: Dict[str, Tuple[float, SongDocument]] = {}
    for ranking, weight in zip(rankings, weights):
        normalized = normalize_ranking(ranking)
        for rank, doc in enumerate(ranking, start=1):
            current_score, existing_doc = weighted_scores.get(doc.song_id, (0.0, doc))
            weighted_scores[doc.song_id] = (
                current_score + weight * (normalized.get(doc.song_id, 0.0) + 1.0 / (RRF_K + rank)),
                existing_doc,
            )
    ordered = sorted(weighted_scores.values(), key=lambda item: (-item[0], item[1].song_id))
    return [doc for _, doc in ordered[:k]]


def expand_query(query: str) -> Dict[str, Any]:
    """Rule-based query expansion used before optional LLM expansion lands."""
    tokens = _tokenize(query)
    keyword_set = {token for token in tokens if len(token) > 2}
    paraphrases = {query}
    for token in list(keyword_set):
        for related in _EXPANSION_HINTS.get(token, []):
            keyword_set.add(related)
            paraphrases.add(f"{query} {related}")
    return {
        "keywords": sorted(keyword_set),
        "queries": sorted(paraphrases),
    }


def embed_texts(
    texts: Sequence[str],
    model: Optional[str] = None,
) -> Dict[str, List[float]]:
    """Embed a batch of texts with Gemini."""
    from google import genai

    client = genai.Client()
    response = client.models.embed_content(
        model=model or DEFAULT_EMBEDDING_MODEL,
        contents=list(texts),
    )
    embeddings = getattr(response, "embeddings", [])
    return {
        text: list(embedding.values)
        for text, embedding in zip(texts, embeddings)
    }


def embed_query(query: str, model: Optional[str] = None) -> List[float]:
    embeddings = embed_texts([query], model=model)
    return embeddings[query]


def save_cache(
    docs: Sequence[SongDocument],
    embeddings: Dict[str, Sequence[float]],
    cache_path: Path,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "documents": [asdict(doc) for doc in docs],
        "embeddings": {key: list(value) for key, value in embeddings.items()},
    }
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_cache(cache_path: Path) -> Dict[str, Any]:
    return json.loads(cache_path.read_text(encoding="utf-8"))


def retrieve(
    query: str,
    docs: Sequence[SongDocument],
    cache_embeddings: Optional[Dict[str, Sequence[float]]] = None,
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    use_llm_expansion: bool = True,
    offline: bool = False,
) -> RetrievalResult:
    expansion = expand_query(query)
    expanded_query = " ".join([query] + expansion["keywords"])

    keyword_ranking = keyword_retrieve(expanded_query, docs, k=k, filters=filters)

    if offline:
        vector_ranking: List[SongDocument] = []
    else:
        query_embedding = embed_query(expanded_query)
        embeddings = dict(cache_embeddings or {})
        missing = [doc for doc in docs if doc.song_id not in embeddings]
        if missing:
            fresh = embed_texts([doc.text for doc in missing])
            for doc in missing:
                embeddings[doc.song_id] = fresh[doc.text]
        vector_ranking = vector_retrieve(
            query_embedding,
            docs,
            embeddings,
            k=k,
            filters=filters,
        )

    prefs = filters or {}
    deterministic = recommend_songs(
        prefs,
        [doc.raw or {} for doc in docs if doc.raw],
        k=k,
        mode=prefs.get("mode", "balanced"),
    )
    deterministic_ranking = [
        docs[[d.song_id for d in docs].index(f"local-{song['id']}")]
        for song, _, _ in deterministic
    ]

    if offline:
        rankings = [keyword_ranking, deterministic_ranking]
        weights = [0.55, 0.45]
    else:
        rankings = [keyword_ranking, vector_ranking, deterministic_ranking]
        weights = [0.35, 0.4, 0.25]
    fused = fuse_rankings(rankings=rankings, weights=weights, k=k)
    return RetrievalResult(
        documents=fused,
        debug={
            "expanded_query": expanded_query,
            "offline": offline,
            "keyword_top": keyword_ranking[0].song_id if keyword_ranking else None,
            "vector_top": vector_ranking[0].song_id if vector_ranking else None,
            "deterministic_top": deterministic_ranking[0].song_id if deterministic_ranking else None,
        },
    )
