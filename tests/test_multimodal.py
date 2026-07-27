import json
from unittest.mock import MagicMock, patch

import pytest

from src.multimodal import (
    VibeProfile,
    analyze_image,
    build_image_prompt,
    validate_image,
    extract_vibe_from_image,
)
from src.ai_service import (
    VibeMatchService,
    RecommendationResult,
)


def test_validate_image_accepts_jpeg():
    assert validate_image(b"\xff\xd8\xff\xe0" + b"\x00" * 100, "image/jpeg") is True


def test_validate_image_accepts_png():
    assert validate_image(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100, "image/png") is True


def test_validate_image_rejects_large_file():
    large = b"\xff\xd8\xff\xe0" + b"\x00" * (10 * 1024 * 1024 + 1)
    assert validate_image(large, "image/jpeg") is False


def test_validate_image_rejects_wrong_mime():
    assert validate_image(b"not an image", "application/pdf") is False


def test_build_image_prompt_includes_schema():
    prompt = build_image_prompt()
    assert "mood" in prompt
    assert "energy" in prompt
    assert "aesthetic" in prompt
    assert "activity" in prompt
    assert "JSON" in prompt


def test_extract_vibe_from_image_parses_gemini_json():
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "mood": "dreamy",
        "energy": 0.4,
        "aesthetic": "soft neon",
        "activity": "evening walk",
        "tags": ["hazy", "urban", "calm"],
    })

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        vibe = extract_vibe_from_image(b"fake image bytes", mime_type="image/jpeg")

    assert isinstance(vibe, VibeProfile)
    assert vibe.mood == "dreamy"
    assert vibe.energy == 0.4
    assert vibe.aesthetic == "soft neon"
    assert "hazy" in vibe.tags


def test_extract_vibe_handles_gemini_failure():
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("API down")

    with patch("google.genai.Client", return_value=mock_client):
        with pytest.raises(RuntimeError, match="Image analysis failed"):
            extract_vibe_from_image(b"fake", mime_type="image/jpeg")


def test_analyze_image_end_to_end_with_mocked_gemini():
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "mood": "energetic",
        "energy": 0.85,
        "aesthetic": "gym",
        "activity": "workout",
        "tags": ["intense", "urban"],
    })

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        vibe = analyze_image(b"\xff\xd8\xff\xe0" + b"\x00" * 100, "image/jpeg")

    assert vibe.mood == "energetic"
    assert vibe.energy == 0.85
    assert vibe.to_query() != ""


def test_vibe_profile_to_query_includes_all_fields():
    vibe = VibeProfile(
        mood="happy",
        energy=0.8,
        aesthetic="sunny pop",
        activity="morning walk",
        tags=["euphoric", "bright"],
    )
    query = vibe.to_query()
    assert "happy" in query
    assert "0.8" in query or "high energy" in query
    assert "sunny pop" in query
    assert "euphoric" in query
