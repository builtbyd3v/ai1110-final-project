import csv
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass

_NUMERIC_FIELDS = (
    "energy",
    "tempo_bpm",
    "valence",
    "danceability",
    "acousticness",
    "popularity",
    "instrumentalness",
)

# Strategy pattern: each mode is a named weight preset. score_song()/Recommender._score()
# look up a preset by name instead of branching on if/elif, so adding a new mode is just
# adding a new dict entry.
BALANCED_WEIGHTS = {"genre": 2.0, "mood": 1.0, "energy": 1.0}
SCORING_MODES: Dict[str, Dict[str, float]] = {
    "balanced": BALANCED_WEIGHTS,
    "genre-first": {"genre": 3.0, "mood": 0.5, "energy": 1.0},
    "mood-first": {"genre": 1.0, "mood": 2.5, "energy": 1.0},
    "energy-focused": {"genre": 1.0, "mood": 0.5, "energy": 2.5},
}

DIVERSITY_ARTIST_PENALTY = 1.5
DIVERSITY_GENRE_PENALTY = 0.75
MAX_SAME_GENRE_BEFORE_PENALTY = 2


def _score_attributes(
    genre_match: bool,
    mood_match: bool,
    energy_diff: float,
    acoustic_bonus: float = 0.0,
    weights: Optional[Dict[str, float]] = None,
    popularity_diff: Optional[float] = None,
    decade_match: Optional[bool] = None,
    mood_tag_overlap: int = 0,
    instrumentalness_diff: Optional[float] = None,
    language_match: Optional[bool] = None,
) -> Tuple[float, List[str]]:
    """Shared point-weighted scoring recipe used by both the OOP and functional paths."""
    weights = weights or BALANCED_WEIGHTS
    score = 0.0
    reasons = []
    if genre_match:
        score += weights["genre"]
        reasons.append(f"genre match (+{weights['genre']:.1f})")
    if mood_match:
        score += weights["mood"]
        reasons.append(f"mood match (+{weights['mood']:.1f})")
    energy_points = (1 - energy_diff) * weights["energy"]
    score += energy_points
    reasons.append(f"energy similarity (+{energy_points:.2f})")
    if acoustic_bonus:
        score += acoustic_bonus
        reasons.append(f"acoustic bonus (+{acoustic_bonus:.2f})")
    if popularity_diff is not None:
        points = (1 - popularity_diff) * 0.5
        score += points
        reasons.append(f"popularity similarity (+{points:.2f})")
    if decade_match:
        score += 0.5
        reasons.append("release decade match (+0.5)")
    if mood_tag_overlap:
        points = 0.5 * mood_tag_overlap
        score += points
        reasons.append(f"mood tag overlap x{mood_tag_overlap} (+{points:.2f})")
    if instrumentalness_diff is not None:
        points = (1 - instrumentalness_diff) * 0.5
        score += points
        reasons.append(f"instrumentalness similarity (+{points:.2f})")
    if language_match:
        score += 0.5
        reasons.append("language match (+0.5)")
    return score, reasons


def _diversify_select(
    candidates: List[Dict[str, Any]],
    k: int,
    get_artist: Callable[[Any], str],
    get_genre: Callable[[Any], str],
) -> List[Dict[str, Any]]:
    """
    Greedy artist/genre diversity re-rank shared by OOP and functional recommenders.

    Each candidate dict must include: item, base_score, reasons.
    Returns selected dicts with adjusted_score and penalty_reasons filled in.
    """
    selected: List[Dict[str, Any]] = []
    artist_counts: Dict[str, int] = {}
    genre_counts: Dict[str, int] = {}
    remaining = [dict(c) for c in candidates]

    while remaining and len(selected) < k:
        for c in remaining:
            artist = get_artist(c["item"])
            genre = get_genre(c["item"])
            penalty = 0.0
            penalty_reasons: List[str] = []
            if artist_counts.get(artist, 0) >= 1:
                penalty += DIVERSITY_ARTIST_PENALTY
                penalty_reasons.append(
                    f"same-artist diversity penalty (-{DIVERSITY_ARTIST_PENALTY:.2f})"
                )
            if genre_counts.get(genre, 0) >= MAX_SAME_GENRE_BEFORE_PENALTY:
                penalty += DIVERSITY_GENRE_PENALTY
                penalty_reasons.append(
                    f"same-genre diversity penalty (-{DIVERSITY_GENRE_PENALTY:.2f})"
                )
            c["adjusted_score"] = c["base_score"] - penalty
            c["penalty_reasons"] = penalty_reasons
        remaining.sort(key=lambda c: c["adjusted_score"], reverse=True)
        best = remaining.pop(0)
        selected.append(best)
        artist = get_artist(best["item"])
        genre = get_genre(best["item"])
        artist_counts[artist] = artist_counts.get(artist, 0) + 1
        genre_counts[genre] = genre_counts.get(genre, 0) + 1

    return selected


@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    popularity: float = 50.0
    release_decade: str = "unknown"
    detailed_mood_tags: str = ""
    instrumentalness: float = 0.0
    language: str = "unknown"
    description: str = ""
    listening_context: str = ""


def song_to_document(song: Any) -> str:
    """Build a searchable text document from a song dict or Song dataclass."""
    if isinstance(song, Song):
        data = {
            "id": song.id,
            "title": song.title,
            "artist": song.artist,
            "genre": song.genre,
            "mood": song.mood,
            "energy": song.energy,
            "detailed_mood_tags": song.detailed_mood_tags,
            "description": song.description,
            "listening_context": song.listening_context,
            "language": song.language,
            "release_decade": song.release_decade,
        }
    else:
        data = song

    tags = str(data.get("detailed_mood_tags") or "").replace(";", ", ")
    parts = [
        f"Title: {data.get('title', '')}",
        f"Artist: {data.get('artist', '')}",
        f"Genre: {data.get('genre', '')}",
        f"Mood: {data.get('mood', '')}",
        f"Energy: {data.get('energy', '')}",
        f"Tags: {tags}",
        f"Description: {data.get('description', '')}",
        f"Listening context: {data.get('listening_context', '')}",
        f"Language: {data.get('language', '')}",
        f"Decade: {data.get('release_decade', '')}",
    ]
    return "\n".join(parts)


@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    target_popularity: Optional[float] = None
    preferred_decade: Optional[str] = None
    mood_tags: Optional[List[str]] = None
    preferred_instrumentalness: Optional[float] = None
    preferred_language: Optional[str] = None


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def _score(self, user: UserProfile, song: Song, mode: str = "balanced") -> Tuple[float, List[str]]:
        """Applies the Algorithm Recipe to one Song/UserProfile pair."""
        acoustic_bonus = 0.5 if user.likes_acoustic and song.acousticness >= 0.6 else 0.0
        return _score_attributes(
            genre_match=song.genre == user.favorite_genre,
            mood_match=song.mood == user.favorite_mood,
            energy_diff=abs(song.energy - user.target_energy),
            acoustic_bonus=acoustic_bonus,
            weights=SCORING_MODES.get(mode, BALANCED_WEIGHTS),
            popularity_diff=abs(song.popularity - user.target_popularity) / 100
            if user.target_popularity is not None else None,
            decade_match=(song.release_decade == user.preferred_decade) if user.preferred_decade else None,
            mood_tag_overlap=len(set(song.detailed_mood_tags.split(";")) & set(user.mood_tags))
            if user.mood_tags else 0,
            instrumentalness_diff=abs(song.instrumentalness - user.preferred_instrumentalness)
            if user.preferred_instrumentalness is not None else None,
            language_match=(song.language == user.preferred_language) if user.preferred_language else None,
        )

    def recommend(self, user: UserProfile, k: int = 5, mode: str = "balanced") -> List[Song]:
        """Returns the top k songs for user, with shared artist/genre diversity re-ranking."""
        candidates = []
        for song in self.songs:
            score, reasons = self._score(user, song, mode)
            candidates.append({"item": song, "base_score": score, "reasons": reasons})
        selected = _diversify_select(
            candidates,
            k=k,
            get_artist=lambda s: s.artist,
            get_genre=lambda s: s.genre,
        )
        return [c["item"] for c in selected]

    def recommend_detailed(
        self, user: UserProfile, k: int = 5, mode: str = "balanced"
    ) -> List[Tuple[Song, float, str]]:
        """Like recommend(), but also returns adjusted score and explanation strings."""
        candidates = []
        for song in self.songs:
            score, reasons = self._score(user, song, mode)
            candidates.append({"item": song, "base_score": score, "reasons": reasons})
        selected = _diversify_select(
            candidates,
            k=k,
            get_artist=lambda s: s.artist,
            get_genre=lambda s: s.genre,
        )
        results = []
        for c in selected:
            explanation = ", ".join(c["reasons"] + c["penalty_reasons"])
            results.append((c["item"], c["adjusted_score"], explanation))
        return results

    def explain_recommendation(self, user: UserProfile, song: Song, mode: str = "balanced") -> str:
        """Returns a human-readable reason string for why song scored the way it did."""
        _, reasons = self._score(user, song, mode)
        return ", ".join(reasons)


def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        songs = list(csv.DictReader(f))
    for song in songs:
        song["id"] = int(song["id"])
        for field_name in _NUMERIC_FIELDS:
            song[field_name] = float(song[field_name])
        song["description"] = song.get("description") or ""
        song["listening_context"] = song.get("listening_context") or ""
    print(f"Loaded songs: {len(songs)}")
    return songs


def score_song(
    user_prefs: Dict, song: Dict, weights: Optional[Dict[str, float]] = None
) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    Required by recommend_songs() and src/main.py
    """
    acoustic_bonus = 0.0
    if user_prefs.get("likes_acoustic") and song.get("acousticness", 0.0) >= 0.6:
        acoustic_bonus = 0.5
    return _score_attributes(
        genre_match=song["genre"] == user_prefs.get("genre"),
        mood_match=song["mood"] == user_prefs.get("mood"),
        energy_diff=abs(song["energy"] - user_prefs.get("energy", 0.0)),
        acoustic_bonus=acoustic_bonus,
        weights=weights,
        popularity_diff=abs(song["popularity"] - user_prefs["target_popularity"]) / 100
        if "target_popularity" in user_prefs else None,
        decade_match=song["release_decade"] == user_prefs["preferred_decade"]
        if "preferred_decade" in user_prefs else None,
        mood_tag_overlap=len(set(song["detailed_mood_tags"].split(";")) & set(user_prefs["mood_tags"]))
        if "mood_tags" in user_prefs else 0,
        instrumentalness_diff=abs(song["instrumentalness"] - user_prefs["preferred_instrumentalness"])
        if "preferred_instrumentalness" in user_prefs else None,
        language_match=song["language"] == user_prefs["preferred_language"]
        if "preferred_language" in user_prefs else None,
    )


def recommend_songs(
    user_prefs: Dict, songs: List[Dict], k: int = 5, mode: str = "balanced"
) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Ranks by score, then greedily re-ranks for artist/genre diversity.
    Required by src/main.py
    """
    weights = SCORING_MODES.get(mode, BALANCED_WEIGHTS)
    candidates = []
    for song in songs:
        score, reasons = score_song(user_prefs, song, weights)
        candidates.append({"item": song, "base_score": score, "reasons": reasons})

    selected = _diversify_select(
        candidates,
        k=k,
        get_artist=lambda s: s["artist"],
        get_genre=lambda s: s["genre"],
    )
    results = []
    for c in selected:
        explanation = ", ".join(c["reasons"] + c["penalty_reasons"])
        results.append((c["item"], c["adjusted_score"], explanation))
    return results
