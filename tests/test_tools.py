from unittest.mock import Mock, patch

import pytest

from src.tools import (
    search_itunes,
    compare_songs,
    get_song_details,
    search_catalog,
    rank_recommendations,
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


def test_search_itunes_returns_normalized_tracks():
    mock_response = Mock()
    mock_response.json.return_value = {
        "resultCount": 2,
        "results": [
            {
                "trackId": 123,
                "trackName": "Blinding Lights",
                "artistName": "The Weeknd",
                "collectionName": "After Hours",
                "artworkUrl100": "https://example.com/art.jpg",
                "previewUrl": "https://example.com/preview.m4a",
                "trackViewUrl": "https://music.apple.com/us/song/123",
                "primaryGenreName": "Pop",
                "releaseDate": "2019-11-29T12:00:00Z",
            },
            {
                "trackId": 456,
                "trackName": "Levitating",
                "artistName": "Dua Lipa",
                "collectionName": "Future Nostalgia",
                "artworkUrl100": None,
                "previewUrl": None,
                "trackViewUrl": "https://music.apple.com/us/song/456",
                "primaryGenreName": "Pop",
                "releaseDate": "2020-03-27T12:00:00Z",
            },
        ],
    }
    mock_response.raise_for_status = Mock()

    with patch("src.tools.requests.get", return_value=mock_response) as mock_get:
        results = search_itunes("pop", limit=2)

    assert len(results) == 2
    assert results[0]["song_id"] == "itunes-123"
    assert results[0]["title"] == "Blinding Lights"
    assert results[0]["artist"] == "The Weeknd"
    assert results[0]["source"] == "itunes"
    assert results[0]["artwork_url"] == "https://example.com/art.jpg"
    mock_get.assert_called_once()
    call_args = mock_get.call_args
    assert "itunes.apple.com" in call_args[0][0]
    assert call_args[1]["params"]["media"] == "music"
    assert call_args[1]["timeout"] == 5


def test_search_itunes_handles_timeout():
    import requests as requests_module

    with patch("src.tools.requests.get", side_effect=requests_module.Timeout("timeout")):
        results = search_itunes("pop")
    assert results == []


def test_search_itunes_handles_empty_results():
    mock_response = Mock()
    mock_response.json.return_value = {"resultCount": 0, "results": []}
    mock_response.raise_for_status = Mock()

    with patch("src.tools.requests.get", return_value=mock_response):
        results = search_itunes("xyznonexistent")
    assert results == []


def test_search_catalog_filters_by_genre(sample_docs):
    results = search_catalog("rainy study music", sample_docs, filters={"genre": "lofi"}, k=3)
    assert len(results) == 1
    assert results[0].song_id == "local-2"


def test_rank_recommendations_returns_scored_pairs(sample_docs):
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
    results = rank_recommendations(user_prefs, sample_docs, k=2, mode="balanced")
    assert len(results) == 2
    song, score, explanation = results[0]
    assert song.title == "Sunrise City"
    assert score > 0
    assert "genre match" in explanation


def test_compare_songs_returns_differences(sample_docs):
    comparison = compare_songs(["local-1", "local-2"], sample_docs)
    assert len(comparison["songs"]) == 2
    assert "energy" in comparison["differences"]
    assert comparison["differences"]["energy"]["local-1"] > comparison["differences"]["energy"]["local-2"]


def test_get_song_details_returns_full_record(sample_docs):
    details = get_song_details("local-1", sample_docs)
    assert details["title"] == "Sunrise City"
    assert details["genre"] == "pop"
    assert details["description"] == "Bright synth-pop."
