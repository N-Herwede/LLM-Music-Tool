"""
Audio analysis and similarity logic.
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .database import db_get_all_tracks, db_get_all_embeddings, db_get_track


# =============================================================================
# AUDIO I/O
# =============================================================================

def load_audio(filepath: str, sr: int = 22050, duration: float = 30.0) -> Tuple[np.ndarray, int]:
    """Load and normalize an audio file."""
    import librosa

    y, sr_orig = librosa.load(filepath, sr=sr, duration=duration, mono=True)

    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    return y, sr


def get_audio_info(filepath: str) -> Dict[str, Any]:
    """Get audio file metadata."""
    import librosa
    import soundfile as sf

    info = sf.info(filepath)
    y, sr = librosa.load(filepath, sr=None, duration=30)

    return {
        "filepath": str(filepath),
        "duration": info.duration,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "format": info.format,
    }


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_features(filepath: str, sr: int = 22050) -> Dict[str, Any]:
    """Extract audio features from a file."""
    import librosa

    y, sr = load_audio(filepath, sr=sr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_arr = np.atleast_1d(tempo)
    tempo_val = float(tempo_arr[0]) if tempo_arr.size else 0.0

    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)

    rms = librosa.feature.rms(y=y)
    energy = float(np.mean(rms))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    return {
        "mfcc_mean": mfcc_mean.tolist(),
        "mfcc_std": mfcc_std.tolist(),
        "tempo_bpm": tempo_val,
        "spectral_centroid": float(np.mean(spec_cent)),
        "spectral_bandwidth": float(np.mean(spec_bw)),
        "spectral_rolloff": float(np.mean(spec_rolloff)),
        "zero_crossing_rate": float(np.mean(zcr)),
        "energy": min(energy * 10, 1.0),
        "valence": float(np.mean(chroma)),
        "chroma_mean": np.mean(chroma, axis=1).tolist(),
    }


def build_feature_vector(filepath: str, dim: int = 64) -> np.ndarray:
    """Build a fixed-size feature vector for similarity comparison."""
    features = extract_features(filepath)

    vector = []
    vector.extend(features["mfcc_mean"])
    vector.extend(features["mfcc_std"])
    vector.append(features["tempo_bpm"] / 200.0)
    vector.append(features["spectral_centroid"] / 5000.0)
    vector.append(features["spectral_bandwidth"] / 3000.0)
    vector.append(features["spectral_rolloff"] / 10000.0)
    vector.append(features["zero_crossing_rate"])
    vector.append(features["energy"])
    vector.extend(features["chroma_mean"])

    arr = np.array(vector, dtype=np.float32)

    if len(arr) < dim:
        arr = np.pad(arr, (0, dim - len(arr)))
    else:
        arr = arr[:dim]

    return arr


# =============================================================================
# SIMILARITY
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def find_similar_tracks(
    db_path: str,
    track_id: int,
    k: int = 5,
    genres: Optional[List[str]] = None,
) -> List[Dict]:
    """Find k most similar tracks to a given track."""
    embeddings = db_get_all_embeddings(db_path)
    tracks = {t["id"]: t for t in db_get_all_tracks(db_path)}

    ref_emb = None
    for tid, emb, meta in embeddings:
        if tid == track_id:
            ref_emb = np.frombuffer(emb, dtype=np.float32)
            break

    if ref_emb is None:
        return []

    similarities = []
    for tid, emb, meta in embeddings:
        if tid == track_id:
            continue

        track = tracks.get(tid, {})

        if genres and track.get("true_genre") not in genres:
            continue

        emb_arr = np.frombuffer(emb, dtype=np.float32)
        sim = cosine_similarity(ref_emb, emb_arr)

        similarities.append(
            {
                "track_id": tid,
                "title": track.get("title"),
                "true_genre": track.get("true_genre"),
                "similarity": sim,
                "tempo_bpm": meta.get("tempo_bpm", 0),
                "energy": meta.get("energy", 0),
            }
        )

    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities[:k]


def compare_tracks(db_path: str, id_a: int, id_b: int) -> Dict:
    """Compare two tracks."""
    embeddings = {t[0]: (t[1], t[2]) for t in db_get_all_embeddings(db_path)}
    tracks = {t["id"]: t for t in db_get_all_tracks(db_path)}

    if id_a not in embeddings or id_b not in embeddings:
        raise ValueError("Track not found")

    emb_a = np.frombuffer(embeddings[id_a][0], dtype=np.float32)
    emb_b = np.frombuffer(embeddings[id_b][0], dtype=np.float32)
    meta_a, meta_b = embeddings[id_a][1], embeddings[id_b][1]
    track_a, track_b = tracks.get(id_a, {}), tracks.get(id_b, {})

    sim = cosine_similarity(emb_a, emb_b)

    if sim >= 0.85:
        level = "Very Similar"
    elif sim >= 0.70:
        level = "Similar"
    elif sim >= 0.50:
        level = "Somewhat Similar"
    else:
        level = "Different"

    same_genre = track_a.get("true_genre") == track_b.get("true_genre")
    explanation = f"These tracks are {level.lower()}. "
    if same_genre:
        explanation += f"Both are {track_a.get('true_genre')}. "
    else:
        explanation += (
            f"Different genres ({track_a.get('true_genre')} vs {track_b.get('true_genre')}). "
        )

    return {
        "track_a": {
            "id": id_a,
            "title": track_a.get("title"),
            "genre": track_a.get("true_genre"),
            "tempo_bpm": meta_a.get("tempo_bpm", 0),
        },
        "track_b": {
            "id": id_b,
            "title": track_b.get("title"),
            "genre": track_b.get("true_genre"),
            "tempo_bpm": meta_b.get("tempo_bpm", 0),
        },
        "similarity_score": sim,
        "similarity_level": level,
        "explanation": explanation,
        "feature_notes": [],
    }


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_library(db_path: str) -> Dict[str, Any]:
    """Analyze the music library statistics."""
    tracks = db_get_all_tracks(db_path)
    embeddings = {t[0]: t[2] for t in db_get_all_embeddings(db_path)}

    if not tracks:
        return {"total_tracks": 0}

    genres = [t["true_genre"] for t in tracks if t.get("true_genre")]
    genre_counts = dict(Counter(genres))
    total = len(tracks)
    genre_pcts = {g: (c / total) * 100 for g, c in genre_counts.items()}

    tempo_by_genre = {}
    energy_by_genre = {}
    tempo_all = []
    energy_all = []

    for track in tracks:
        genre = track.get("true_genre")
        tid = track["id"]
        if genre and tid in embeddings:
            meta = embeddings[tid]
            tempo = meta.get("tempo_bpm", 0)
            energy = meta.get("energy", 0)

            if genre not in tempo_by_genre:
                tempo_by_genre[genre] = []
                energy_by_genre[genre] = []

            tempo_by_genre[genre].append(tempo)
            energy_by_genre[genre].append(energy)
            tempo_all.append(tempo)
            energy_all.append(energy)

    avg_tempo = {g: np.mean(v) for g, v in tempo_by_genre.items()}
    avg_energy = {g: np.mean(v) for g, v in energy_by_genre.items()}

    all_tempos = [t for temps in tempo_by_genre.values() for t in temps]
    tempo_stats = {
        "mean": np.mean(all_tempos) if all_tempos else 0,
        "median": float(np.median(all_tempos)) if all_tempos else 0,
        "std": np.std(all_tempos) if all_tempos else 0,
        "min": np.min(all_tempos) if all_tempos else 0,
        "max": np.max(all_tempos) if all_tempos else 0,
        "p25": float(np.percentile(all_tempos, 25)) if all_tempos else 0,
        "p75": float(np.percentile(all_tempos, 75)) if all_tempos else 0,
    }

    energy_stats = {
        "mean": float(np.mean(energy_all)) if energy_all else 0,
        "median": float(np.median(energy_all)) if energy_all else 0,
        "std": float(np.std(energy_all)) if energy_all else 0,
        "min": float(np.min(energy_all)) if energy_all else 0,
        "max": float(np.max(energy_all)) if energy_all else 0,
        "p25": float(np.percentile(energy_all, 25)) if energy_all else 0,
        "p75": float(np.percentile(energy_all, 75)) if energy_all else 0,
    }

    track_map = {t["id"]: t for t in tracks}
    tempo_rows = [
        {
            "id": tid,
            "tempo_bpm": float(meta.get("tempo_bpm", 0)),
        }
        for tid, meta in embeddings.items()
        if meta.get("tempo_bpm") is not None
    ]
    energy_rows = [
        {
            "id": tid,
            "energy": float(meta.get("energy", 0)),
        }
        for tid, meta in embeddings.items()
        if meta.get("energy") is not None
    ]

    def _with_track_info(rows: List[Dict], key: str, reverse: bool) -> Dict[str, Any]:
        if not rows:
            return {}
        rows = sorted(rows, key=lambda x: x.get(key, 0), reverse=reverse)
        top = rows[0]
        track = track_map.get(top["id"], {})
        return {
            "id": top["id"],
            "title": track.get("title"),
            "genre": track.get("true_genre"),
            key: top.get(key, 0),
        }

    outliers = {
        "fastest": _with_track_info(tempo_rows, "tempo_bpm", True),
        "slowest": _with_track_info(tempo_rows, "tempo_bpm", False),
        "most_energetic": _with_track_info(energy_rows, "energy", True),
        "least_energetic": _with_track_info(energy_rows, "energy", False),
    }

    top_genres = [
        {"genre": g, "count": c, "pct": genre_pcts.get(g, 0)}
        for g, c in sorted(genre_counts.items(), key=lambda x: -x[1])
    ]

    coverage = {
        "tracks_with_embeddings": len(embeddings),
        "tracks_missing_embeddings": max(total - len(embeddings), 0),
    }

    return {
        "total_tracks": total,
        "genre_distribution": genre_counts,
        "genre_percentages": genre_pcts,
        "avg_tempo_by_genre": avg_tempo,
        "avg_energy_by_genre": avg_energy,
        "tempo_stats": tempo_stats,
        "energy_stats": energy_stats,
        "top_genres": top_genres,
        "outliers": outliers,
        "coverage": coverage,
    }


def infer_key_from_features(features: Dict[str, Any]) -> str:
    """Infer a rough key from chroma and valence."""
    chroma = features.get("chroma_mean") or []
    if not chroma:
        return "Unknown"
    pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    idx = int(np.argmax(chroma))
    tonic = pitch_classes[idx % 12]
    mode = "major" if features.get("valence", 0.0) >= 0.5 else "minor"
    return f"{tonic} {mode}"


def infer_mood_from_features(features: Dict[str, Any]) -> str:
    """Infer a simple mood label from tempo, energy, and valence."""
    tempo = float(features.get("tempo_bpm", 0))
    energy = float(features.get("energy", 0))
    valence = float(features.get("valence", 0))

    if energy >= 0.7 and tempo >= 120:
        return "energetic"
    if energy <= 0.4 and tempo <= 110:
        return "calm"
    if valence >= 0.6:
        return "happy"
    if valence <= 0.4:
        return "sad"
    return "balanced"


def predict_genre_for_file(db_path: str, filepath: str, k: int = 5) -> Dict[str, Any]:
    """Predict genre for a file using similarity to the library."""
    vec = build_feature_vector(filepath)
    features = extract_features(filepath)
    tracks = {t["id"]: t for t in db_get_all_tracks(db_path)}

    sims = []
    for tid, emb, meta in db_get_all_embeddings(db_path):
        track = tracks.get(tid, {})
        genre = track.get("true_genre")
        if not genre:
            continue
        emb_arr = np.frombuffer(emb, dtype=np.float32)
        sim = cosine_similarity(vec, emb_arr)
        sims.append(
            {
                "id": tid,
                "title": track.get("title"),
                "genre": genre,
                "sim": sim,
                "tempo_bpm": float(meta.get("tempo_bpm", 0)) if meta else None,
                "energy": float(meta.get("energy", 0)) if meta else None,
            }
        )

    sims.sort(key=lambda x: x["sim"], reverse=True)
    top = sims[:k]

    if not top:
        return {
            "predicted_genre": None,
            "confidence": 0.0,
            "top_matches": [],
            "features": features,
            "reason": "No reference tracks with genre available.",
        }

    genre_scores: Dict[str, float] = {}
    for row in top:
        genre_scores[row["genre"]] = genre_scores.get(row["genre"], 0.0) + row["sim"]

    best_genre = max(genre_scores.items(), key=lambda x: x[1])[0]
    total_score = sum(genre_scores.values()) or 1.0
    confidence = genre_scores[best_genre] / total_score

    reason = (
        f"Top matches cluster in {best_genre} "
        f"({sum(1 for r in top if r['genre'] == best_genre)}/{len(top)})."
    )

    return {
        "predicted_genre": best_genre,
        "confidence": confidence,
        "top_matches": top,
        "features": {
            "tempo_bpm": features.get("tempo_bpm", 0),
            "energy": features.get("energy", 0),
            "spectral_centroid": features.get("spectral_centroid", 0),
            "valence": features.get("valence", 0),
        },
        "reason": reason,
    }
