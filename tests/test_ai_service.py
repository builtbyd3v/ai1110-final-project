import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.ai_service import (
    MAX_TOOL_CALLS,
    RecommendationResult,
    VibeMatchService,
    build_system_prompt,
    build_grounding_prompt,
)
from src.rag import SongDocument


@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch):
    """Retrieval must never call the real embedding API in tests (CI has no key)."""
    monkeypatch.setattr("src.rag.embed_query", lambda query, model=None: [1.0, 0.0])
    monkeypatch.setattr(
        "src.rag.embed_texts",
        lambda texts, model=None: {t: [1.0, 0.0] for t in texts},
    )


@pytest.fixture
def sample_docs():
    return [
        SongDocument(
            song_id="local-1",
            title="Sunrise City",
            artist="Neon Echo",
            source="local",
            text="...",
            description="Bright synth-pop.",
            listening_context="morning",
            genre="pop",
            mood="happy",
            energy=0.82,
            raw={"id": 1, "title": "Sunrise City", "artist": "Neon Echo", "genre": "pop", "mood": "happy", "energy": 0.82, "popularity": 82.0, "release_decade": "2020s", "detailed_mood_tags": "euphoric", "instrumentalness": 0.02, "language": "english"},
        ),
        SongDocument(
            song_id="local-2",
            title="Library Rain",
            artist="Paper Lanterns",
            source="local",
            text="...",
            description="Rainy lo-fi.",
            listening_context="studying",
            genre="lofi",
            mood="chill",
            energy=0.35,
            raw={"id": 2, "title": "Library Rain", "artist": "Paper Lanterns", "genre": "lofi", "mood": "chill", "energy": 0.35, "popularity": 45.0, "release_decade": "2020s", "detailed_mood_tags": "cozy", "instrumentalness": 0.9, "language": "instrumental"},
        ),
    ]


def test_build_system_prompt_includes_grounding_rules():
    prompt = build_system_prompt()
    assert "ONLY" in prompt
    assert "cite" in prompt.lower() or "citation" in prompt.lower()
    assert "catalog" in prompt.lower()
    assert "hallucinate" in prompt.lower() or "invent" in prompt.lower()


def test_build_grounding_prompt_includes_evidence(sample_docs):
    prompt = build_grounding_prompt(
        user_query="happy pop for morning",
        local_docs=sample_docs[:1],
        itunes_results=[],
        user_prefs={"genre": "pop", "mood": "happy", "energy": 0.8},
    )
    assert "Sunrise City" in prompt
    assert "Neon Echo" in prompt
    assert "pop" in prompt
    assert "0.82" in prompt


def test_service_recommend_returns_structured_result(sample_docs):
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "recommendations": [
            {
                "title": "Sunrise City",
                "artist": "Neon Echo",
                "reason": "Matches happy pop request with bright synth energy",
                "source": "local",
                "evidence": "genre match, mood match",
            }
        ],
        "summary": "Found upbeat pop for your morning.",
    })

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    service = VibeMatchService(docs=sample_docs, client=mock_client)
    result = service.recommend(
        query="happy pop for morning",
        user_prefs={"genre": "pop", "mood": "happy", "energy": 0.8},
        k=3,
    )

    assert isinstance(result, RecommendationResult)
    assert result.recommendations[0]["title"] == "Sunrise City"
    assert result.recommendations[0]["source"] == "local"
    assert result.summary != ""
    assert result.fallback_used is False


def test_service_falls_back_to_deterministic_on_gemini_failure(sample_docs):
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("API down")

    service = VibeMatchService(docs=sample_docs, client=mock_client)
    result = service.recommend(
        query="happy pop",
        user_prefs={"genre": "pop", "mood": "happy", "energy": 0.8},
        k=2,
    )

    assert result.fallback_used is True
    assert len(result.recommendations) == 2
    assert result.recommendations[0]["title"] == "Sunrise City"
    assert "error" in result.metadata or "warning" in result.metadata


def test_service_separates_local_and_itunes_sources(sample_docs):
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "recommendations": [
            {"title": "Sunrise City", "artist": "Neon Echo", "reason": "Local match", "source": "local", "evidence": "genre"},
            {"title": "Blinding Lights", "artist": "The Weeknd", "reason": "Similar vibe", "source": "itunes", "evidence": "live search"},
        ],
        "summary": "Mixed local and live results.",
    })

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("src.ai_service.search_itunes") as mock_itunes:
        mock_itunes.return_value = [
            {"song_id": "itunes-1", "title": "Blinding Lights", "artist": "The Weeknd", "source": "itunes"}
        ]
        service = VibeMatchService(docs=sample_docs, client=mock_client)
        result = service.recommend(
            query="synth pop",
            user_prefs={},
            k=2,
            include_live=True,
        )

    sources = {r["source"] for r in result.recommendations}
    assert "local" in sources
    assert "itunes" in sources
    assert result.metadata["local_count"] >= 1
    assert result.metadata["itunes_count"] >= 1


def test_service_rejects_unsupported_titles(sample_docs):
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "recommendations": [
            {"title": "Made Up Song", "artist": "Fake Artist", "reason": "Hallucinated", "source": "local", "evidence": "none"},
        ],
        "summary": "Bad output.",
    })

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    service = VibeMatchService(docs=sample_docs, client=mock_client)
    result = service.recommend(query="test", user_prefs={}, k=2)

    # Hallucinated titles should be filtered out
    titles = [r["title"] for r in result.recommendations]
    assert "Made Up Song" not in titles


def _final_answer_response(payload):
    response = MagicMock()
    response.text = json.dumps(payload)
    response.function_calls = []
    return response


def test_tool_loop_executes_search_itunes_and_cites_result(sample_docs):
    tool_call_response = MagicMock()
    tool_call_response.text = ""
    tool_call_response.function_calls = [
        SimpleNamespace(name="search_itunes", args={"query": "synth pop", "limit": 2})
    ]
    tool_call_response.candidates = [SimpleNamespace(content="tool-call-content")]

    final_response = _final_answer_response({
        "recommendations": [
            {"title": "Blinding Lights", "artist": "The Weeknd", "reason": "Live discovery", "source": "itunes", "evidence": "itunes search for synth pop"},
        ],
        "summary": "One live discovery.",
    })

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [tool_call_response, final_response]

    itunes_items = [{
        "song_id": "itunes-1",
        "title": "Blinding Lights",
        "artist": "The Weeknd",
        "source": "itunes",
        "artwork_url": "https://example.com/art.jpg",
        "preview_url": "https://example.com/preview.m4a",
        "track_view_url": "https://music.apple.com/song/1",
    }]
    with patch("src.ai_service.search_itunes", return_value=itunes_items):
        service = VibeMatchService(docs=sample_docs, client=mock_client)
        result = service.recommend(query="synth pop", user_prefs={}, k=3, offline=True)

    assert result.fallback_used is False
    assert result.recommendations[0]["title"] == "Blinding Lights"
    assert result.recommendations[0]["source"] == "itunes"
    # Enrichment: artwork and preview URLs from the tool result
    assert result.recommendations[0]["artwork_url"] == "https://example.com/art.jpg"
    assert result.recommendations[0]["preview_url"] == "https://example.com/preview.m4a"
    assert result.metadata["tool_calls"] == ["search_itunes"]


def test_tool_loop_search_catalog_uses_local_docs(sample_docs):
    tool_call_response = MagicMock()
    tool_call_response.text = ""
    tool_call_response.function_calls = [
        SimpleNamespace(name="search_catalog", args={"query": "rainy lo-fi", "limit": 2})
    ]
    tool_call_response.candidates = [SimpleNamespace(content="tool-call-content")]

    final_response = _final_answer_response({
        "recommendations": [
            {"title": "Library Rain", "artist": "Paper Lanterns", "reason": "Rainy lo-fi match", "source": "local", "evidence": "catalog search for rainy lo-fi"},
        ],
        "summary": "Found it via catalog search.",
    })

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [tool_call_response, final_response]

    service = VibeMatchService(docs=sample_docs, client=mock_client)
    result = service.recommend(query="rainy lo-fi", user_prefs={}, k=3, offline=True)

    assert result.recommendations[0]["title"] == "Library Rain"
    assert result.metadata["tool_calls"] == ["search_catalog"]


def test_tool_loop_stops_at_max_calls_and_falls_back(sample_docs):
    def make_tool_response():
        response = MagicMock()
        response.text = ""
        response.function_calls = [
            SimpleNamespace(name="search_catalog", args={"query": "loop", "limit": 1})
        ]
        response.candidates = [SimpleNamespace(content="tool-call-content")]
        return response

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [
        make_tool_response() for _ in range(MAX_TOOL_CALLS)
    ]

    service = VibeMatchService(docs=sample_docs, client=mock_client)
    result = service.recommend(query="loop forever", user_prefs={}, k=2, offline=True)

    assert result.fallback_used is True
    assert result.metadata.get("tool_loop_exhausted") is True
    assert len(result.metadata["tool_calls"]) == MAX_TOOL_CALLS
    assert mock_client.models.generate_content.call_count == MAX_TOOL_CALLS


def test_service_loads_embedding_cache_when_available(sample_docs, tmp_path):
    cache_file = tmp_path / "embeddings.json"
    cache_file.write_text(json.dumps({
        "documents": [],
        "embeddings": {"local-1": [1.0, 0.0], "local-2": [0.0, 1.0]},
    }), encoding="utf-8")

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _final_answer_response({
        "recommendations": [], "summary": "none"
    })

    service = VibeMatchService(
        docs=sample_docs, client=mock_client, cache_path=str(cache_file)
    )
    assert service.cache_embeddings == {"local-1": [1.0, 0.0], "local-2": [0.0, 1.0]}
