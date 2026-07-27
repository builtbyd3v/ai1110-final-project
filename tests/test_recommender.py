from src.recommender import (
    Song,
    UserProfile,
    Recommender,
    load_songs,
    recommend_songs,
    song_to_document,
    SCORING_MODES,
)


def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_diversity_penalizes_same_artist_on_oop_path():
    """OOP and functional paths must share artist diversity penalties."""
    songs = [
        Song(
            id=1,
            title="Pop A",
            artist="Same Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.1,
        ),
        Song(
            id=2,
            title="Pop B",
            artist="Same Artist",
            genre="pop",
            mood="happy",
            energy=0.79,
            tempo_bpm=118,
            valence=0.88,
            danceability=0.8,
            acousticness=0.1,
        ),
        Song(
            id=3,
            title="Indie C",
            artist="Other Artist",
            genre="pop",
            mood="happy",
            energy=0.78,
            tempo_bpm=122,
            valence=0.85,
            danceability=0.8,
            acousticness=0.2,
        ),
    ]
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    results = Recommender(songs).recommend(user, k=2)
    titles = [song.title for song in results]

    assert titles[0] == "Pop A"
    # Without diversity, Pop B would be #2. With shared diversity, Indie C wins.
    assert titles[1] == "Indie C"


def test_recommend_songs_applies_same_artist_diversity():
    songs = [
        {
            "id": 1,
            "title": "Pop A",
            "artist": "Same Artist",
            "genre": "pop",
            "mood": "happy",
            "energy": 0.8,
            "popularity": 50.0,
            "release_decade": "2020s",
            "detailed_mood_tags": "euphoric",
            "instrumentalness": 0.0,
            "language": "english",
        },
        {
            "id": 2,
            "title": "Pop B",
            "artist": "Same Artist",
            "genre": "pop",
            "mood": "happy",
            "energy": 0.79,
            "popularity": 50.0,
            "release_decade": "2020s",
            "detailed_mood_tags": "euphoric",
            "instrumentalness": 0.0,
            "language": "english",
        },
        {
            "id": 3,
            "title": "Indie C",
            "artist": "Other Artist",
            "genre": "pop",
            "mood": "happy",
            "energy": 0.78,
            "popularity": 50.0,
            "release_decade": "2020s",
            "detailed_mood_tags": "euphoric",
            "instrumentalness": 0.0,
            "language": "english",
        },
    ]
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
    results = recommend_songs(prefs, songs, k=3)
    titles = [song["title"] for song, _, _ in results]

    assert titles[:2] == ["Pop A", "Indie C"]
    assert titles[2] == "Pop B"
    assert "same-artist diversity penalty" in results[2][2]


def test_all_scoring_modes_are_recognized():
    assert set(SCORING_MODES) == {
        "balanced",
        "genre-first",
        "mood-first",
        "energy-focused",
    }


def test_load_songs_includes_description_fields():
    songs = load_songs("data/songs.csv")
    assert len(songs) == 18
    first = songs[0]
    assert first["description"].strip()
    assert first["listening_context"].strip()
    assert "Sunrise" in first["title"] or first["id"] == 1


def test_song_to_document_includes_searchable_text():
    song = {
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
    }
    doc = song_to_document(song)
    assert "Sunrise City" in doc
    assert "Neon Echo" in doc
    assert "synth-pop" in doc
    assert "morning commute" in doc
    assert "euphoric" in doc
