"""
Audio tagging with a pretrained model.
"""

from pathlib import Path
from typing import Any, Dict, List
import numpy as np


def tag_audio(filepath: str, top_k: int = 5) -> Dict[str, Any]:
    """Return top tags with scores using musicnn."""
    try:
        from musicnn.extractor import extractor
    except ImportError as exc:
        raise RuntimeError("musicnn is not installed. Add it to requirements.txt") from exc

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    taggram, tags, _features = extractor(str(path), model="MTT_musicnn", extract_features=False)
    tag_scores = np.mean(taggram, axis=0)
    pairs = sorted(zip(tags, tag_scores), key=lambda x: x[1], reverse=True)
    top = pairs[: max(int(top_k), 1)]
    return {
        "tags": [{"tag": t, "score": float(s)} for t, s in top],
        "model": "MTT_musicnn",
    }
