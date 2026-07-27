import json
from unittest.mock import MagicMock, patch

import pytest

from src.ai_service import (
    RecommendationResult,
    VibeMatchService,
    build_system_prompt,
    build_grounding_prompt,
)
from src.rag import SongDocument


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
