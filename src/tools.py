from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from src.rag import SongDocument, keyword_retrieve
from src.recommender import recommend_songs

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
REQUEST_TIMEOUT = 5


def search_itunes(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search iTunes for live track discoveries. Returns normalized song dicts."""
    try:
        response = requests.get(
            ITUNES_SEARCH_URL,
            params={
                "term": query,
                "media": "music",
                "entity": "song",
                "limit": limit,
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException:
        return []

    results = []
    for item in data.get("results", []):
        results.append(
            {
                "song_id": f"itunes-{item.get('trackId', '')}",
                "title": item.get("trackName", "Unknown"),
                "artist": item.get("artistName", "Unknown"),
                "album": item.get("collectionName"),
                "genre": item.get("primaryGenreName", ""),
                "artwork_url": item.get("artworkUrl100"),
                "preview_url": item.get("previewUrl"),
                "track_view_url": item.get("trackViewUrl"),
                "release_date": item.get("releaseDate"),
                "source": "itunes",
            }
        )
    return results


def search_catalog(
    query: str,
    docs: Sequence[SongDocument],
    filters: Optional[Dict[str, Any]] = None,
    k: int = 5,
) -> List[SongDocument]:
    """Keyword search over local catalog documents."""
    return keyword_retrieve(query, docs, k=k, filters=filters)


def rank_recommendations(
    user_prefs: Dict[str, Any],
    docs: Sequence[SongDocument],
    k: int = 5,
    mode: str = "balanced",
) -> List[Tuple[SongDocument, float, str]]:
    """Deterministic re-rank of catalog docs using the original scoring recipe."""
    songs = [doc.raw for doc in docs if doc.raw]
    ranked = recommend_songs(user_prefs, songs, k=k, mode=mode)
    by_id = {f"local-{song['id']}": doc for doc, song in zip(docs, songs)}
    results = []
    for song, score, explanation in ranked:
        doc = by_id.get(f"local-{song['id']}")
        if doc:
            results.append((doc, score, explanation))
    return results


def compare_songs(
    song_ids: Sequence[str],
    docs: Sequence[SongDocument],
) -> Dict[str, Any]:
    """Compare selected songs on key attributes."""
    selected = [doc for doc in docs if doc.song_id in song_ids]
    return {
        "songs": [
            {
                "song_id": doc.song_id,
                "title": doc.title,
                "artist": doc.artist,
                "genre": doc.genre,
                "mood": doc.mood,
                "energy": doc.energy,
            }
            for doc in selected
        ],
        "differences": {
            "energy": {doc.song_id: doc.energy for doc in selected},
            "genre": {doc.song_id: doc.genre for doc in selected},
            "mood": {doc.song_id: doc.mood for doc in selected},
        },
    }


def get_song_details(
    song_id: str,
    docs: Sequence[SongDocument],
) -> Dict[str, Any]:
    """Return the full local catalog record for one song."""
    for doc in docs:
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
                "raw": doc.raw or {},
            }
    return {}
