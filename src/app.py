from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

from src.ai_service import VibeMatchService
from src.multimodal import analyze_image, validate_image
from src.rag import load_documents_from_csv
from src.recommender import SCORING_MODES

load_dotenv()

st.set_page_config(
    page_title="VibeMatch",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "docs" not in st.session_state:
    st.session_state.docs = load_documents_from_csv("data/songs.csv")
if "service" not in st.session_state:
    st.session_state.service = VibeMatchService(docs=st.session_state.docs)

service: VibeMatchService = st.session_state.service

# ---------------------------------------------------------------------------
# Sidebar: structured preferences + system health
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🎵 VibeMatch")
    st.markdown("*AI-powered music discovery*")

    st.divider()
    st.subheader("Preferences")
    genre = st.selectbox(
        "Genre",
        ["", "pop", "lofi", "rock", "ambient", "jazz", "synthwave", "hip-hop",
         "classical", "folk", "metal", "r&b", "country", "punk", "electronic"],
        index=0,
    )
    mood = st.selectbox(
        "Mood",
        ["", "happy", "chill", "intense", "relaxed", "moody", "focused",
         "peaceful", "nostalgic", "dreamy", "energetic", "sad", "uplifting",
         "rebellious", "aggressive"],
        index=0,
    )
    energy = st.slider("Energy", 0.0, 1.0, 0.5, 0.05)
    mode = st.selectbox("Scoring mode", list(SCORING_MODES.keys()), index=0)
    k = st.slider("Results", 1, 10, 5)
    include_live = st.checkbox("Include live iTunes discoveries", value=False)

    user_prefs: Dict[str, Any] = {"energy": energy}
    if genre:
        user_prefs["genre"] = genre
    if mood:
        user_prefs["mood"] = mood

    st.divider()
    st.subheader("System health")
    api_key_ok = bool(os.getenv("GEMINI_API_KEY"))
    st.write("🔑 Gemini API:", "✅ Connected" if api_key_ok else "❌ Missing key")
    st.write("📚 Catalog:", f"✅ {len(st.session_state.docs)} songs loaded")
    st.write("🌐 iTunes:", "✅ Enabled" if include_live else "⏸️ Disabled")

    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("VibeMatch")
st.caption("Describe a vibe, upload an image, or use the sidebar controls.")

tab_text, tab_image = st.tabs(["💬 Text request", "📷 Match this vibe"])

with tab_text:
    query = st.text_area(
        "What are you in the mood for?",
        placeholder="e.g. chill lo-fi for late night studying, or energetic pop for a morning run",
        height=100,
    )
    col1, col2 = st.columns([1, 4])
    with col1:
        search_clicked = st.button("Get recommendations", type="primary")
    with col2:
        st.caption("Powered by hybrid RAG + Gemini. Local catalog first, live iTunes optional.")

    if search_clicked and query.strip():
        with st.spinner("Finding your vibe..."):
            try:
                result = service.recommend(
                    query=query,
                    user_prefs=user_prefs,
                    k=k,
                    mode=mode,
                    include_live=include_live,
                )
                st.session_state.history.append({
                    "type": "text",
                    "query": query,
                    "result": result,
                })
            except Exception as exc:
                st.error(f"Recommendation failed: {exc}")

with tab_image:
    uploaded = st.file_uploader(
        "Upload an image (JPEG, PNG, or WebP under 5MB)",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded:
        image_bytes = uploaded.read()
        mime = uploaded.type or "image/jpeg"
        if not validate_image(image_bytes, mime):
            st.error("Invalid image. Use JPEG, PNG, or WebP under 5MB.")
        else:
            st.image(image_bytes, width=300, caption="Uploaded image")
            if st.button("Analyze vibe", type="primary"):
                with st.spinner("Reading the room..."):
                    try:
                        vibe = analyze_image(image_bytes, mime)
                        st.success(
                            f"Detected: **{vibe.mood}** mood, **{vibe.energy:.1f}** energy, "
                            f"**{vibe.aesthetic}** aesthetic, **{vibe.activity}** activity"
                        )
                        result = service.recommend(
                            query=vibe.to_query(),
                            user_prefs={**user_prefs, "mood": vibe.mood, "energy": vibe.energy},
                            k=k,
                            mode=mode,
                            include_live=include_live,
                        )
                        st.session_state.history.append({
                            "type": "image",
                            "query": vibe.to_query(),
                            "vibe": vibe,
                            "result": result,
                        })
                    except Exception as exc:
                        st.error(f"Image analysis failed: {exc}")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if st.session_state.history:
    st.divider()
    latest = st.session_state.history[-1]
    result = latest["result"]

    if result.fallback_used:
        st.warning("Gemini unavailable — showing deterministic local recommendations.")

    st.subheader("Recommendations")
    st.caption(result.summary)

    local_recs = [r for r in result.recommendations if r.get("source") == "local"]
    itunes_recs = [r for r in result.recommendations if r.get("source") == "itunes"]

    if local_recs:
        st.markdown("**From your catalog**")
        for rec in local_recs:
            with st.container(border=True):
                cols = st.columns([3, 1])
                with cols[0]:
                    st.markdown(f"**{rec['title']}** by {rec['artist']}")
                    st.caption(rec["reason"])
                    if rec.get("evidence"):
                        st.caption(f"Evidence: {rec['evidence']}")
                with cols[1]:
                    st.markdown(f"`{rec['source']}`")

    if itunes_recs:
        st.markdown("**Live discoveries (iTunes)**")
        for rec in itunes_recs:
            with st.container(border=True):
                cols = st.columns([3, 1])
                with cols[0]:
                    st.markdown(f"**{rec['title']}** by {rec['artist']}")
                    st.caption(rec["reason"])
                    if rec.get("evidence"):
                        st.caption(f"Evidence: {rec['evidence']}")
                with cols[1]:
                    st.markdown(f"`{rec['source']}`")

    with st.expander("Debug info"):
        st.json(result.metadata)

# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
if len(st.session_state.history) > 1:
    with st.expander("Recent requests"):
        for i, item in enumerate(reversed(st.session_state.history[:-1]), start=1):
            kind = "📷" if item["type"] == "image" else "💬"
            st.markdown(f"{i}. {kind} `{item['query'][:80]}`")
