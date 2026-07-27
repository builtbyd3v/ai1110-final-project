import json
from pathlib import Path

import pytest

from src.rag import (
    SongDocument,
    build_documents,
    expand_query,
    fuse_rankings,
    keyword_retrieve,
    load_documents_from_csv,
    normalize_ranking,
    reciprocal_rank_fusion,
    retrieve,
    save_cache,
    load_cache,
    vector_retrieve,
)


@pytest.fixture
def sample_songs():
    return [
        {
            "id": 1,
            "title": "Sunrise City",
            "artist": "Neon Echo",
            "genre": "pop",
            "mood": "happy",
            "energy": 0.82,
            "detailed_mood_tags": "euphoric;energetic",
            "description": "Bright synth-pop for sunny mornings.",
            "listening_context": "morning commute; workouts",
            "language": "english",
            "release_decade": "2020s",
        },
        {
            "id": 2,
            "title": "Library Rain",
            "artist": "Paper Lanterns",
            "genre": "lofi",
            "mood": "chill",
            "energy": 0.35,
            "detailed_mood_tags": "cozy;melancholic",
            "description": "Gentle rainy-day lo-fi for quiet indoor hours.",
            "listening_context": "studying; winding down",
            "language": "instrumental",
            "release_decade": "2020s",
        },
        {
            "id": 3,
            "title": "Storm Runner",
            "artist": "Voltline",
            "genre": "rock",
            "mood": "intense",
            "energy": 0.91,
            "detailed_mood_tags": "aggressive;driving",
            "description": "Driving rock track for intense focus.",
            "listening_context": "gym; running",
            "language": "english",
            "release_decade": "2010s",
        },
    ]


def test_build_documents_creates_stable_ids_and_text(sample_songs):
    docs = build_documents(sample_songs)
    assert len(docs) == 3
    assert docs[0].song_id == "local-1"
    assert "Sunrise City" in docs[0].text
    assert docs[0].source == "local"
    assert docs[0].title == "Sunrise City"
    assert docs[0].artist == "Neon Echo"


def test_keyword_retrieve_matches_description_terms(sample_songs):
    docs = build_documents(sample_songs)
    results = keyword_retrieve("quiet rainy study lo-fi", docs, k=2)
    assert len(results) == 2
    assert results[0].song_id == "local-2"
    assert results[0].title == "Library Rain"


def test_keyword_retrieve_respects_genre_filter(sample_songs):
    docs = build_documents(sample_songs)
    results = keyword_retrieve("music", docs, k=3, filters={"genre": "rock"})
    assert len(results) == 1
    assert results[0].song_id == "local-3"


def test_normalize_ranking_scores_first_result_highest():
    docs = [
        SongDocument(song_id="a", title="A", artist="X", source="local", text="a"),
        SongDocument(song_id="b", title="B", artist="Y", source="local", text="b"),
        SongDocument(song_id="c", title="C", artist="Z", source="local", text="c"),
    ]
    scores = normalize_ranking(docs)
    assert scores["a"] == 1.0
    assert scores["b"] == 0.5
    assert scores["c"] == 0.0


def test_reciprocal_rank_fusion_combines_multiple_rankings(sample_songs):
    docs = build_documents(sample_songs)
    keyword = [docs[0], docs[1], docs[2]]
    vector = [docs[1], docs[2], docs[0]]
    deterministic = [docs[0], docs[2], docs[1]]
    fused = reciprocal_rank_fusion(
        [keyword, vector, deterministic], k=3
    )
    assert fused[0].song_id == "local-1"
    assert set(d.song_id for d in fused) == {"local-1", "local-2", "local-3"}


def test_fuse_rankings_supports_weighted_lists(sample_songs):
    docs = build_documents(sample_songs)
    keyword = [docs[0], docs[1]]
    vector = [docs[2], docs[1]]
    fused = fuse_rankings(
        rankings=[keyword, vector],
        weights=[0.7, 0.3],
        k=2,
    )
    assert fused[0].song_id == "local-1"


def test_load_documents_from_csv_reads_current_catalog():
    docs = load_documents_from_csv("data/songs.csv")
    assert len(docs) == 18
    first = docs[0]
    assert first.song_id.startswith("local-")
    assert first.description
    assert first.listening_context


def test_expand_query_returns_terms_and_paraphrases():
    expanded = expand_query("sad music for a rainy night")
    assert "rainy" in expanded["keywords"]
    assert "sad" in expanded["keywords"]
    assert any("melancholic" in q or "late-night" in q for q in expanded["queries"])


def test_cache_roundtrip_preserves_embeddings(tmp_path: Path):
    docs = [
        SongDocument(
            song_id="local-1",
            title="A",
            artist="X",
            source="local",
            text="hello",
        )
    ]
    embeddings = {"local-1": [0.1, 0.2, 0.3]}
    cache_file = tmp_path / "embeddings.json"
    save_cache(docs, embeddings, cache_file)
    loaded = load_cache(cache_file)
    assert loaded["documents"][0]["song_id"] == "local-1"
    assert loaded["embeddings"]["local-1"] == [0.1, 0.2, 0.3]


def test_vector_retrieve_orders_by_cosine_similarity(sample_songs):
    docs = build_documents(sample_songs)
    embeddings = {
        "local-1": [1.0, 0.0],
        "local-2": [0.0, 1.0],
        "local-3": [0.9, 0.1],
    }
    query_embedding = [0.95, 0.05]
    results = vector_retrieve(query_embedding, docs, embeddings, k=2)
    assert results[0].song_id == "local-1"
    assert results[1].song_id == "local-3"


def test_retrieve_combines_keyword_vector_and_deterministic(sample_songs, monkeypatch):
    docs = build_documents(sample_songs)
    monkeypatch.setattr("src.rag.embed_texts", lambda texts, model=None: {t: [1.0, 0.0] for t in texts})
    monkeypatch.setattr(
        "src.rag.embed_query", lambda query, model=None: [1.0, 0.0]
    )
    result = retrieve(
        query="bright upbeat pop",
        docs=docs,
        cache_embeddings={
            "local-1": [1.0, 0.0],
            "local-2": [0.0, 1.0],
            "local-3": [0.5, 0.5],
        },
        k=2,
        use_llm_expansion=False,
    )
    assert result.documents[0].song_id == "local-1"
    assert result.debug["keyword_top"] == "local-1"
