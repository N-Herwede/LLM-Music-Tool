"""
YouTube download helpers.
"""

from pathlib import Path
from typing import Any, Dict, Tuple


def download_youtube_audio(url: str, output_dir: Path) -> Tuple[str, Dict[str, Any]]:
    """Download a YouTube audio track as WAV and return filepath + metadata."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp is not installed. Add it to requirements.txt") from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "outtmpl": str(output_dir / "%(title)s.%(id)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if isinstance(info, dict) and "entries" in info:
            info = info["entries"][0]
        base_path = ydl.prepare_filename(info)

    wav_path = str(Path(base_path).with_suffix(".wav"))
    return wav_path, info
