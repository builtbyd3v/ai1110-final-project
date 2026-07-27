import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from src.rag import DEFAULT_GENERATION_MODEL

MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_MIMES = {"image/jpeg", "image/png", "image/webp"}


@dataclass
class VibeProfile:
    mood: str
    energy: float
    aesthetic: str
    activity: str
    tags: List[str] = field(default_factory=list)

    def to_query(self) -> str:
        parts = [self.mood, self.aesthetic, self.activity]
        parts.extend(self.tags)
        energy_desc = "high energy" if self.energy > 0.6 else "low energy" if self.energy < 0.4 else "moderate energy"
        parts.append(energy_desc)
        return " ".join(p for p in parts if p)


def validate_image(data: bytes, mime_type: str) -> bool:
    if len(data) > MAX_IMAGE_BYTES:
        return False
    if mime_type not in ALLOWED_MIMES:
        return False
    if mime_type == "image/jpeg" and not data.startswith(b"\xff\xd8\xff"):
        return False
    if mime_type == "image/png" and not data.startswith(b"\x89PNG"):
        return False
    return True


def build_image_prompt() -> str:
    return (
        "Analyze this image and extract its musical vibe. "
        "Return ONLY valid JSON with these exact keys: "
        "mood (single word), energy (float 0.0-1.0), "
        "aesthetic (2-3 words describing visual style), "
        "activity (what someone doing this would listen to), "
        "tags (array of 3-5 descriptive strings). "
        "No markdown, no explanation, just JSON."
    )


def extract_vibe_from_image(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    model: Optional[str] = None,
) -> VibeProfile:
    client = genai.Client()
    try:
        response = client.models.generate_content(
            model=model or DEFAULT_GENERATION_MODEL,
            contents=[
                build_image_prompt(),
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        return VibeProfile(
            mood=data.get("mood", "unknown"),
            energy=float(data.get("energy", 0.5)),
            aesthetic=data.get("aesthetic", ""),
            activity=data.get("activity", ""),
            tags=data.get("tags", []),
        )
    except Exception as exc:
        raise RuntimeError(f"Image analysis failed: {exc}") from exc


def analyze_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> VibeProfile:
    if not validate_image(image_bytes, mime_type):
        raise ValueError("Invalid or unsupported image. Use JPEG, PNG, or WebP under 5MB.")
    return extract_vibe_from_image(image_bytes, mime_type)
