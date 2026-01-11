"""
Text-to-speech helpers (offline).
"""

from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    idx = 1
    while True:
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def text_to_speech(
    text: str,
    output_dir: Optional[str] = None,
    filename: Optional[str] = None,
    rate: Optional[int] = None,
    voice: Optional[str] = None,
) -> Dict[str, Any]:
    """Render text to a WAV file using pyttsx3."""
    try:
        import pyttsx3
    except ImportError as exc:
        raise RuntimeError("pyttsx3 is not installed. Add it to requirements.txt") from exc

    if not text or not text.strip():
        raise ValueError("Text is empty")

    out_dir = Path(output_dir) if output_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    if filename:
        fname = Path(filename).name
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"tts_{stamp}.wav"

    output_path = _unique_path(out_dir / fname)

    engine = pyttsx3.init()
    if rate is not None:
        engine.setProperty("rate", int(rate))
    if voice:
        engine.setProperty("voice", voice)

    engine.save_to_file(text, str(output_path))
    engine.runAndWait()

    return {
        "output_path": str(output_path),
    }
