from src.app import build_user_prefs


def test_build_user_prefs_includes_all_fields():
    prefs = build_user_prefs("pop", "happy", 0.8)
    assert prefs == {"genre": "pop", "mood": "happy", "energy": 0.8}


def test_build_user_prefs_omits_empty_selections():
    prefs = build_user_prefs("", "chill", 0.4)
    assert "genre" not in prefs
    assert prefs["mood"] == "chill"
    assert prefs["energy"] == 0.4


def test_build_user_prefs_defaults_to_energy_only():
    prefs = build_user_prefs("", "", 0.5)
    assert prefs == {"energy": 0.5}
