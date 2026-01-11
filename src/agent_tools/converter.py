"""
Audio conversion helpers.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import subprocess


def _normalize_format(fmt: str) -> str:
    fmt = fmt.strip().lower()
    if fmt.startswith("."):
        fmt = fmt[1:]
    if not fmt or not fmt.isalnum():
        raise ValueError("Invalid output format")
    return fmt


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


def convert_audio(
    filepath: str,
    output_format: str,
    output_dir: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Convert a local audio file using ffmpeg."""
    input_path = Path(filepath)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    fmt = _normalize_format(output_format)
    out_dir = Path(output_dir) if output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = out_dir / f"{input_path.stem}.{fmt}"
    if not overwrite:
        output_path = _unique_path(output_path)

    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-vn",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is not installed or not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() if exc.stderr else str(exc)
        raise RuntimeError(f"ffmpeg failed: {detail}") from exc

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "format": fmt,
        "overwrote": overwrite,
    }
