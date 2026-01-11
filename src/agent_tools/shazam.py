"""
Shazam identification helpers.
"""

from pathlib import Path
from typing import Any, Dict
import asyncio


async def _identify_async(filepath: str) -> Dict[str, Any]:
    try:
        from shazamio import Shazam
    except ImportError as exc:
        raise RuntimeError("shazamio is not installed. Add it to requirements.txt") from exc

    shazam = Shazam()
    return await shazam.recognize(str(filepath))


def identify_track(filepath: str) -> Dict[str, Any]:
    """Identify a track from an audio file using Shazam."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    result = asyncio.run(_identify_async(str(path)))
    track = result.get("track") or {}
    return {
        "title": track.get("title"),
        "artist": track.get("subtitle"),
        "album": (track.get("sections") or [{}])[0].get("metadata", [{}])[0].get("text"),
        "genres": (track.get("genres") or {}).get("primary"),
        "image": (track.get("images") or {}).get("coverart"),
        "shazam_url": (track.get("share") or {}).get("href"),
    }
