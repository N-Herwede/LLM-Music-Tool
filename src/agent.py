"""
Music Agent
===========

AI agent that orchestrates LLM and audio analysis tools.
"""

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from agent_tools.api_clients import LLM, get_llm
from agent_tools.analysis import (
    analyze_library,
    build_feature_vector,
    extract_features,
    compare_tracks,
    cosine_similarity,
    find_similar_tracks,
    predict_genre_for_file,
    infer_key_from_features,
    infer_mood_from_features,
)
from agent_tools.database import (
    db_add_track,
    db_get_all_embeddings,
    db_get_all_tracks,
    db_get_track,
    db_store_embedding,
    db_store_external_tracks,
    db_store_external_as_tracks,
    db_update_embedding_metadata,
    db_create_playlist,
    db_list_playlists,
    db_get_playlist,
    db_add_track_to_playlist,
    db_remove_track_from_playlist,
    db_delete_playlist,
)
from agent_tools.reports import generate_report, generate_track_report
from agent_tools.visualization import create_visualizations, create_waveform_image, create_spectrogram_image
from agent_tools.youtube import download_youtube_audio
from agent_tools.converter import convert_audio
from agent_tools.shazam import identify_track
from agent_tools.deezer import get_music_trends
from agent_tools.tagger import tag_audio
from agent_tools.tts import text_to_speech


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "music_library.db"
REPORTS_DIR = ROOT / "data" / "processed" / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"
YOUTUBE_DIR = ROOT / "data" / "raw" / "youtube"
CONVERT_DIR = ROOT / "data" / "processed" / "converted"
TTS_DIR = ROOT / "data" / "processed" / "tts"
SAVED_UPLOAD_DIR = ROOT / "data" / "processed" / "uploads"
UPLOAD_DIR = ROOT / "data" / "uploads"

MOODS = {
    "blues": ["sad", "melancholic", "relaxed"],
    "classical": ["calm", "peaceful", "focused"],
    "country": ["nostalgic", "warm", "relaxed"],
    "disco": ["happy", "energetic", "party"],
    "hiphop": ["energetic", "confident", "powerful"],
    "jazz": ["relaxed", "smooth", "creative"],
    "metal": ["aggressive", "powerful", "intense"],
    "pop": ["happy", "upbeat", "fun"],
    "reggae": ["relaxed", "chill", "peaceful"],
    "rock": ["energetic", "rebellious", "exciting"],
}


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

TOOLS = [
    {"type": "function", "function": {"name": "analyze_library", "description": "Analyze library", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "recommend_tracks", "description": "Find similar tracks", "parameters": {"type": "object", "properties": {"track_id": {"type": "integer"}, "k": {"type": "integer"}, "genres": {"type": "array", "items": {"type": "string"}}}, "required": ["track_id"]}}},
    {"type": "function", "function": {"name": "compare_tracks", "description": "Compare two tracks", "parameters": {"type": "object", "properties": {"track_a": {"type": "integer"}, "track_b": {"type": "integer"}}, "required": ["track_a", "track_b"]}}},
    {"type": "function", "function": {"name": "find_by_mood", "description": "Recommend by mood", "parameters": {"type": "object", "properties": {"mood": {"type": "string"}, "k": {"type": "integer"}}, "required": ["mood"]}}},
    {"type": "function", "function": {"name": "generate_playlist", "description": "Generate playlist", "parameters": {"type": "object", "properties": {"theme": {"type": "string"}, "k": {"type": "integer"}}, "required": ["theme"]}}},
    {"type": "function", "function": {"name": "filter_by_tempo", "description": "Search by BPM", "parameters": {"type": "object", "properties": {"min_bpm": {"type": "number"}, "max_bpm": {"type": "number"}}, "required": ["min_bpm", "max_bpm"]}}},
    {"type": "function", "function": {"name": "generate_report", "description": "Generate a report for a track or the full library", "parameters": {"type": "object", "properties": {"scope": {"type": "string"}, "track_id": {"type": "integer"}, "filepath": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "download_youtube_audio", "description": "Download audio from YouTube and ingest into the database", "parameters": {"type": "object", "properties": {"url": {"type": "string"}, "genre": {"type": "string"}, "artist": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {"name": "generate_waveform", "description": "Generate waveform image for a track or file", "parameters": {"type": "object", "properties": {"track_id": {"type": "integer"}, "filepath": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "get_track_info", "description": "Get track info", "parameters": {"type": "object", "properties": {"track_id": {"type": "integer"}}, "required": ["track_id"]}}},
    {"type": "function", "function": {"name": "analyze_upload", "description": "Analyze uploaded audio", "parameters": {"type": "object", "properties": {"filepath": {"type": "string"}}, "required": ["filepath"]}}},
    {"type": "function", "function": {"name": "save_upload", "description": "Save an uploaded audio file to the database (with optional Shazam metadata)", "parameters": {"type": "object", "properties": {"filepath": {"type": "string"}}, "required": ["filepath"]}}},
    {"type": "function", "function": {"name": "convert_audio", "description": "Convert a local audio file to another format", "parameters": {"type": "object", "properties": {"filepath": {"type": "string"}, "output_format": {"type": "string"}, "output_dir": {"type": "string"}, "overwrite": {"type": "boolean"}}, "required": ["filepath", "output_format"]}}},
    {"type": "function", "function": {"name": "identify_track", "description": "Identify a track from a local audio file", "parameters": {"type": "object", "properties": {"filepath": {"type": "string"}}, "required": ["filepath"]}}},
    {"type": "function", "function": {"name": "get_music_trends", "description": "Get trending music from Deezer (charts or genres)", "parameters": {"type": "object", "properties": {"country": {"type": "string"}, "limit": {"type": "integer"}, "chart": {"type": "string"}, "genre": {"type": "string"}, "save": {"type": "boolean"}}, "required": []}}},
    {"type": "function", "function": {"name": "detect_genre", "description": "Predict the genre/style of a local audio file", "parameters": {"type": "object", "properties": {"filepath": {"type": "string"}, "k": {"type": "integer"}}, "required": ["filepath"]}}},
    {"type": "function", "function": {"name": "tag_audio", "description": "Predict audio tags (mood/genre/instrument) for a local file", "parameters": {"type": "object", "properties": {"filepath": {"type": "string"}, "top_k": {"type": "integer"}}, "required": ["filepath"]}}},
    {"type": "function", "function": {"name": "text_to_speech", "description": "Convert text to speech and save audio", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "filename": {"type": "string"}, "rate": {"type": "integer"}, "voice": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "tag_track", "description": "Tag an existing track and store AI tags in the database", "parameters": {"type": "object", "properties": {"track_id": {"type": "integer"}, "filepath": {"type": "string"}, "top_k": {"type": "integer"}}, "required": []}}},
    {"type": "function", "function": {"name": "create_playlist", "description": "Create a playlist", "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "description": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "list_playlists", "description": "List playlists", "parameters": {"type": "object", "properties": {}}, "required": []}},
    {"type": "function", "function": {"name": "get_playlist", "description": "Get playlist details and ordered tracks", "parameters": {"type": "object", "properties": {"playlist_id": {"type": "integer"}}, "required": ["playlist_id"]}}},
    {"type": "function", "function": {"name": "add_to_playlist", "description": "Append a track to a playlist", "parameters": {"type": "object", "properties": {"playlist_id": {"type": "integer"}, "track_id": {"type": "integer"}}, "required": ["playlist_id", "track_id"]}}},
    {"type": "function", "function": {"name": "remove_from_playlist", "description": "Remove a track from a playlist by position", "parameters": {"type": "object", "properties": {"playlist_id": {"type": "integer"}, "position": {"type": "integer"}}, "required": ["playlist_id", "position"]}}},
    {"type": "function", "function": {"name": "delete_playlist", "description": "Delete a playlist", "parameters": {"type": "object", "properties": {"playlist_id": {"type": "integer"}}, "required": ["playlist_id"]}}},
]

TOOL_ALIASES = {
    "analyze": "analyze_library",
    "recommend": "recommend_tracks",
    "compare": "compare_tracks",
    "mood": "find_by_mood",
    "playlist": "generate_playlist",
    "tempo": "filter_by_tempo",
    "report": "generate_report",
    "youtube_download": "download_youtube_audio",
    "audio_preview": "generate_waveform",
    "track": "get_track_info",
    "upload": "analyze_upload",
    "shazam_identify": "identify_track",
    "music_trends": "get_music_trends",
}


# =============================================================================
# TOOL EXECUTION
# =============================================================================

def _generate_library_narrative(
    llm: Optional[LLM],
    stats: Dict[str, Any],
    highlights: Dict[str, Dict],
) -> Optional[str]:
    if not llm:
        return None
    payload = {
        "total_tracks": stats.get("total_tracks", 0),
        "top_genres": stats.get("top_genres", [])[:5],
        "tempo_stats": stats.get("tempo_stats", {}),
        "energy_stats": stats.get("energy_stats", {}),
        "highlights": {
            k: {
                "title": v.get("title"),
                "genre": v.get("genre"),
                "tempo_bpm": v.get("tempo_bpm"),
                "energy": v.get("energy"),
            }
            for k, v in (highlights or {}).items()
        },
    }
    system = (
        "You are a music data analyst. Write a concise 3-5 sentence narrative summary. "
        "Use only the provided numbers. No bullets."
    )
    user = f"Library stats JSON:\n{json.dumps(payload, indent=2)}"
    resp = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
    text = (resp.get("content") or "").strip()
    return text or None


def _generate_track_narrative(
    llm: Optional[LLM],
    track: Dict[str, Any],
    features: Dict[str, Any],
) -> Optional[str]:
    if not llm:
        return None
    payload = {
        "title": track.get("title"),
        "artist": track.get("artist"),
        "genre": track.get("genre"),
        "tempo_bpm": round(float(track.get("tempo_bpm", 0)), 1),
        "energy": round(float(track.get("energy", 0)), 2),
        "key": track.get("key"),
        "mood": track.get("mood"),
        "valence": round(float(features.get("valence", 0)), 2),
        "spectral_centroid": round(float(features.get("spectral_centroid", 0)), 0),
    }
    system = (
        "You are a music analyst. Write 1-2 sentences of commentary. "
        "Use only the provided fields; do not add new facts."
    )
    user = f"Track data JSON:\n{json.dumps(payload, indent=2)}"
    resp = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
    text = (resp.get("content") or "").strip()
    return text or None
def _collect_tracks(
    db_path: str,
    k: int = 5,
    genres: Optional[List[str]] = None,
    min_bpm: Optional[float] = None,
    max_bpm: Optional[float] = None,
    min_energy: Optional[float] = None,
    max_energy: Optional[float] = None,
) -> List[Dict]:
    tracks = {t["id"]: t for t in db_get_all_tracks(db_path)}
    results = []
    for tid, emb, meta in db_get_all_embeddings(db_path):
        track = tracks.get(tid, {})
        genre = track.get("true_genre")
        if genres and genre not in genres:
            continue
        tempo = float(meta.get("tempo_bpm", 0))
        energy = float(meta.get("energy", 0))
        if min_bpm is not None and tempo < min_bpm:
            continue
        if max_bpm is not None and tempo > max_bpm:
            continue
        if min_energy is not None and energy < min_energy:
            continue
        if max_energy is not None and energy > max_energy:
            continue
        results.append(
            {
                "id": tid,
                "title": track.get("title"),
                "genre": genre,
                "tempo_bpm": round(tempo, 1),
                "energy": round(energy, 2),
            }
        )
    random.shuffle(results)
    return results[:k]


def _similar_by_vector(db_path: str, vec: np.ndarray, k: int = 5) -> List[Dict]:
    tracks = {t["id"]: t for t in db_get_all_tracks(db_path)}
    sims = []
    for tid, emb, meta in db_get_all_embeddings(db_path):
        emb_arr = np.frombuffer(emb, dtype=np.float32)
        sim = cosine_similarity(vec, emb_arr)
        track = tracks.get(tid, {})
        sims.append(
            {
                "id": tid,
                "title": track.get("title"),
                "genre": track.get("true_genre"),
                "sim": round(sim, 3),
                "tempo_bpm": round(float(meta.get("tempo_bpm", 0)), 1),
                "energy": round(float(meta.get("energy", 0)), 2),
            }
        )
    sims.sort(key=lambda x: x["sim"], reverse=True)
    return sims[:k]


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


def run_tool(
    name: str,
    args: Dict[str, Any],
    db: Optional[str] = None,
    llm: Optional[LLM] = None,
) -> Dict:
    db = db or str(DB_PATH)
    args = args or {}
    name = TOOL_ALIASES.get(name, name)

    try:
        def _validate_local_audio_path(path_value: Optional[str]) -> Optional[str]:
            if not path_value:
                return "Missing filepath"
            if str(path_value).startswith("deezer:"):
                return "Track is from Deezer and has no local audio file."
            return None

        if name == "analyze_library":
            stats = analyze_library(db)
            return {
                "ok": True,
                "total": stats["total_tracks"],
                "genres": stats["genre_distribution"],
                "top_genres": stats.get("top_genres", []),
                "coverage": stats.get("coverage", {}),
                "tempo_stats": stats.get("tempo_stats", {}),
                "energy_stats": stats.get("energy_stats", {}),
                "tempo_by_genre": {k: round(v, 1) for k, v in stats.get("avg_tempo_by_genre", {}).items()},
                "energy_by_genre": {k: round(v, 2) for k, v in stats.get("avg_energy_by_genre", {}).items()},
                "outliers": stats.get("outliers", {}),
            }

        if name == "recommend_tracks":
            tid = args.get("track_id", 1)
            k = args.get("k", 5)
            recs = find_similar_tracks(db, tid, k, args.get("genres"))
            ref = db_get_track(db, tid)
            return {
                "ok": True,
                "ref": {"id": tid, "title": ref.get("title") if ref else None},
                "tracks": [
                    {
                        "id": r["track_id"],
                        "title": r.get("title"),
                        "genre": r.get("true_genre"),
                        "sim": round(r["similarity"], 3),
                    }
                    for r in recs
                ],
            }

        if name == "compare_tracks":
            r = compare_tracks(db, args["track_a"], args["track_b"])
            return {
                "ok": True,
                "sim": round(r["similarity_score"], 3),
                "level": r["similarity_level"],
                "note": r["explanation"],
            }

        if name == "find_by_mood":
            mood = str(args.get("mood", "")).lower()
            genres = [g for g, moods in MOODS.items() if mood in moods]
            tracks = _collect_tracks(db, k=args.get("k", 5), genres=genres) if genres else []
            return {"ok": True, "tracks": tracks}

        if name == "generate_playlist":
            theme = str(args.get("theme", "")).lower()
            k = args.get("k", 8)
            if any(w in theme for w in ["workout", "sport", "gym"]):
                tracks = _collect_tracks(db, k=k, min_bpm=120, min_energy=0.7)
            elif any(w in theme for w in ["relax", "chill", "sleep"]):
                tracks = _collect_tracks(db, k=k, max_bpm=110, max_energy=0.4)
            elif any(w in theme for w in ["study", "focus"]):
                tracks = _collect_tracks(db, k=k, max_bpm=120, max_energy=0.5)
            elif "party" in theme:
                tracks = _collect_tracks(db, k=k, min_energy=0.7)
            else:
                tracks = _collect_tracks(db, k=k)
            return {"ok": True, "tracks": tracks}

        if name == "filter_by_tempo":
            tracks = _collect_tracks(
                db,
                k=args.get("k", 10),
                min_bpm=args.get("min_bpm"),
                max_bpm=args.get("max_bpm"),
            )
            return {"ok": True, "tracks": tracks}

        if name == "get_track_info":
            t = db_get_track(db, args.get("track_id", 1))
            if not t:
                return {"ok": False, "error": "Not found"}
            embs = {e[0]: e[2] for e in db_get_all_embeddings(db)}
            m = embs.get(t["id"], {})
            return {
                "ok": True,
                "id": t["id"],
                "title": t.get("title"),
                "genre": t.get("true_genre"),
                "tempo": round(m.get("tempo_bpm", 0), 1),
                "energy": round(m.get("energy", 0), 2),
            }

        if name == "generate_report":
            scope = (args.get("scope") or "").lower().strip()
            track_id = args.get("track_id")
            filepath = args.get("filepath")
            if scope in {"library", "full"} or (not track_id and not filepath):
                stats = analyze_library(db)
                plots = create_visualizations(stats, str(PLOTS_DIR))
                tracks = {t["id"]: t for t in db_get_all_tracks(db)}
                embeddings = {t[0]: t[2] for t in db_get_all_embeddings(db)}

                track_rows = []
                for tid, track in tracks.items():
                    meta = embeddings.get(tid, {})
                    track_rows.append(
                        {
                            "id": tid,
                            "title": track.get("title"),
                            "artist": track.get("artist"),
                            "genre": track.get("true_genre"),
                            "filepath": track.get("filepath"),
                            "tempo_bpm": float(meta.get("tempo_bpm", 0)) if meta else None,
                            "energy": float(meta.get("energy", 0)) if meta else None,
                        }
                    )

                def _best(items, key, default=None):
                    items = [i for i in items if i.get(key) is not None]
                    if not items:
                        return default
                    return max(items, key=lambda x: x.get(key, 0))

                def _worst(items, key, default=None):
                    items = [i for i in items if i.get(key) is not None]
                    if not items:
                        return default
                    return min(items, key=lambda x: x.get(key, 0))

                highlights = {
                    "Most Energetic": _best(track_rows, "energy") or {},
                    "Fastest Tempo": _best(track_rows, "tempo_bpm") or {},
                    "Slowest Tempo": _worst(track_rows, "tempo_bpm") or {},
                }
                narrative = _generate_library_narrative(llm, stats, highlights)

                track_sections = []
                for tid, track in list(tracks.items())[:3]:
                    meta = embeddings.get(tid, {})
                    filepath = track.get("filepath")
                    if not filepath:
                        continue
                    if not Path(filepath).exists():
                        continue
                    out_dir = PLOTS_DIR / "tracks"
                    try:
                        wave = create_waveform_image(filepath, str(out_dir))
                        spec = create_spectrogram_image(filepath, str(out_dir))
                    except Exception:
                        continue
                    track_sections.append(
                        {
                            "track": {
                                "id": tid,
                                "title": track.get("title"),
                                "artist": track.get("artist"),
                                "genre": track.get("true_genre"),
                                "tempo_bpm": float(meta.get("tempo_bpm", 0)) if meta else None,
                                "energy": float(meta.get("energy", 0)) if meta else None,
                            },
                            "images": {"Waveform": wave, "Mel Spectrogram": spec},
                        }
                    )

                report_paths = generate_report(
                    stats,
                    plots,
                    str(REPORTS_DIR),
                    highlights=highlights,
                    track_sections=track_sections,
                    narrative=narrative,
                )
                report_path = report_paths.get("markdown_path")
                html_path = report_paths.get("html_path")
                pdf_path = report_paths.get("pdf_path")
                report_url = None
                html_url = None
                pdf_url = None
                try:
                    rel_md = Path(report_path).resolve().relative_to(REPORTS_DIR.resolve())
                    report_url = f"/reports/{rel_md.as_posix()}"
                except Exception:
                    report_url = None
                try:
                    rel_html = Path(html_path).resolve().relative_to(REPORTS_DIR.resolve())
                    html_url = f"/reports/{rel_html.as_posix()}"
                except Exception:
                    html_url = None
                if pdf_path:
                    try:
                        rel_pdf = Path(pdf_path).resolve().relative_to(REPORTS_DIR.resolve())
                        pdf_url = f"/reports/{rel_pdf.as_posix()}"
                    except Exception:
                        pdf_url = None
                return {
                    "ok": True,
                    "scope": "library",
                    "report_path": report_path,
                    "report_url": report_url,
                    "html_path": html_path,
                    "html_url": html_url,
                    "pdf_path": pdf_path,
                    "pdf_url": pdf_url,
                    "plots": [str(Path(p).name) for p in plots],
                    "total": stats.get("total_tracks", 0),
                }

            track = None
            meta = {}
            if track_id:
                track = db_get_track(db, int(track_id))
                if not track:
                    return {"ok": False, "error": "Track not found"}
                filepath = track.get("filepath")
                emb_map = {e[0]: e[2] for e in db_get_all_embeddings(db)}
                meta = emb_map.get(track["id"], {})
            if not filepath:
                return {"ok": False, "error": "Missing track_id or filepath"}
            err = _validate_local_audio_path(filepath)
            if err:
                return {"ok": False, "error": err}

            features = extract_features(filepath)
            key = infer_key_from_features(features)
            mood = infer_mood_from_features(features)
            out_dir = PLOTS_DIR / "tracks"
            wave = create_waveform_image(filepath, str(out_dir))
            spec = create_spectrogram_image(filepath, str(out_dir))

            track_info = {
                "id": track.get("id") if track else None,
                "title": track.get("title") if track else Path(filepath).stem,
                "artist": track.get("artist") if track else None,
                "genre": track.get("true_genre") if track else None,
                "tempo_bpm": float(meta.get("tempo_bpm", features.get("tempo_bpm", 0))),
                "energy": float(meta.get("energy", features.get("energy", 0))),
                "ai_tags": meta.get("ai_tags") if meta else [],
                "key": key,
                "mood": mood,
            }
            summary = (
                f"{track_info['title']} feels {mood} with a {key} tonality. "
                f"Tempo is {track_info['tempo_bpm']:.1f} BPM and energy is {track_info['energy']:.2f}."
            )
            narrative = _generate_track_narrative(llm, track_info, features)

            report_paths = generate_track_report(
                track_info,
                features,
                {"Waveform": wave, "Mel Spectrogram": spec},
                str(REPORTS_DIR),
                summary=summary,
                narrative=narrative,
            )
            report_path = report_paths.get("markdown_path")
            html_path = report_paths.get("html_path")
            pdf_path = report_paths.get("pdf_path")
            report_url = None
            html_url = None
            pdf_url = None
            try:
                rel_md = Path(report_path).resolve().relative_to(REPORTS_DIR.resolve())
                report_url = f"/reports/{rel_md.as_posix()}"
            except Exception:
                report_url = None
            try:
                rel_html = Path(html_path).resolve().relative_to(REPORTS_DIR.resolve())
                html_url = f"/reports/{rel_html.as_posix()}"
            except Exception:
                html_url = None
            if pdf_path:
                try:
                    rel_pdf = Path(pdf_path).resolve().relative_to(REPORTS_DIR.resolve())
                    pdf_url = f"/reports/{rel_pdf.as_posix()}"
                except Exception:
                    pdf_url = None
            return {
                "ok": True,
                "scope": "track",
                "report_path": report_path,
                "report_url": report_url,
                "html_path": html_path,
                "html_url": html_url,
                "pdf_path": pdf_path,
                "pdf_url": pdf_url,
            }

        if name == "download_youtube_audio":
            url = args.get("url")
            if not url:
                return {"ok": False, "error": "Missing url"}
            genre = args.get("genre") or "unknown"
            artist = args.get("artist")

            filepath, info = download_youtube_audio(url, YOUTUBE_DIR)
            title = info.get("title") if isinstance(info, dict) else None
            if not title:
                title = Path(filepath).stem
            if not artist and isinstance(info, dict):
                artist = info.get("uploader") or info.get("channel")

            track_id = db_add_track(db, title, filepath, genre, artist)
            if track_id is None:
                return {"ok": False, "error": "Failed to insert track"}

            features = extract_features(filepath)
            embedding = build_feature_vector(filepath)
            metadata = {
                "tempo_bpm": features.get("tempo_bpm", 0),
                "energy": features.get("energy", 0),
                "valence": features.get("valence", 0),
                "spectral_centroid": features.get("spectral_centroid", 0),
                "source": "youtube",
                "source_url": url,
            }
            if isinstance(info, dict) and info.get("duration") is not None:
                metadata["duration"] = info["duration"]

            db_store_embedding(db, track_id, embedding, metadata)
            download_url = None
            try:
                out_path = Path(filepath).resolve()
                youtube_root = YOUTUBE_DIR.resolve()
                rel = out_path.relative_to(youtube_root)
                download_url = f"/downloads/youtube/{rel.as_posix()}"
            except ValueError:
                download_url = None
            return {
                "ok": True,
                "track_id": track_id,
                "title": title,
                "artist": artist,
                "genre": genre,
                "filepath": filepath,
                "download_url": download_url,
            }

        if name == "generate_waveform":
            filepath = args.get("filepath")
            if not filepath:
                track_id = args.get("track_id")
                if track_id:
                    track = db_get_track(db, track_id)
                    filepath = track.get("filepath") if track else None
            if not filepath:
                return {"ok": False, "error": "Missing filepath or track_id"}
            err = _validate_local_audio_path(filepath)
            if err:
                return {"ok": False, "error": err}
            out_dir = PLOTS_DIR / "waveforms"
            img_path = create_waveform_image(filepath, str(out_dir))
            image_url = None
            try:
                rel = Path(img_path).resolve().relative_to(REPORTS_DIR.resolve())
                image_url = f"/reports/{rel.as_posix()}"
            except Exception:
                image_url = None
            return {"ok": True, "image_path": img_path, "image_url": image_url}

        if name == "analyze_upload":
            filepath = args.get("filepath")
            err = _validate_local_audio_path(filepath)
            if err:
                return {"ok": False, "error": err}
            vec = build_feature_vector(filepath)
            tracks = _similar_by_vector(db, vec, k=args.get("k", 5))
            return {"ok": True, "tracks": tracks}

        if name == "save_upload":
            filepath = args.get("filepath")
            if not filepath:
                return {"ok": False, "error": "Missing filepath"}
            input_path = Path(filepath)
            if not input_path.exists():
                return {"ok": False, "error": f"File not found: {filepath}"}
            SAVED_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            dest_path = _unique_path(SAVED_UPLOAD_DIR / input_path.name)
            try:
                input_path.replace(dest_path)
            except Exception:
                dest_path.write_bytes(input_path.read_bytes())
                input_path.unlink(missing_ok=True)

            info = {}
            try:
                info = identify_track(str(dest_path)) or {}
            except Exception:
                info = {}

            title = info.get("title") or dest_path.stem
            artist = info.get("artist")
            genre = None
            genres = info.get("genres")
            if isinstance(genres, list) and genres:
                genre = genres[0]
            elif isinstance(genres, str):
                genre = genres
            genre = genre or "unknown"

            track_id = db_add_track(db, title, str(dest_path), genre, artist=artist)
            if track_id is None:
                return {"ok": False, "error": "Failed to save track"}

            features = extract_features(str(dest_path))
            embedding = build_feature_vector(str(dest_path))
            metadata = {
                "tempo_bpm": features.get("tempo_bpm", 0),
                "energy": features.get("energy", 0),
                "valence": features.get("valence", 0),
                "spectral_centroid": features.get("spectral_centroid", 0),
                "source": "upload",
                "shazam_url": info.get("shazam_url"),
                "album": info.get("album"),
                "genres": info.get("genres"),
            }
            db_store_embedding(db, int(track_id), embedding, metadata)
            return {
                "ok": True,
                "track_id": int(track_id),
                "title": title,
                "artist": artist,
                "genre": genre,
                "filepath": str(dest_path),
            }

        if name == "convert_audio":
            filepath = args.get("filepath")
            output_format = args.get("output_format") or args.get("format")
            if not filepath or not output_format:
                return {"ok": False, "error": "Missing filepath or output_format"}
            output_dir = args.get("output_dir") or str(CONVERT_DIR)
            try:
                input_path = Path(filepath).resolve()
                upload_root = UPLOAD_DIR.resolve()
                input_path.relative_to(upload_root)
                output_dir = str(CONVERT_DIR)
            except ValueError:
                pass
            overwrite = bool(args.get("overwrite", False))
            result = convert_audio(filepath, output_format, output_dir, overwrite)
            download_url = None
            try:
                out_path = Path(result["output_path"]).resolve()
                convert_root = CONVERT_DIR.resolve()
                rel = out_path.relative_to(convert_root)
                download_url = f"/download/{rel.as_posix()}"
            except ValueError:
                download_url = None
            return {
                "ok": True,
                "input_path": result["input_path"],
                "output_path": result["output_path"],
                "format": result["format"],
                "download_url": download_url,
            }

        if name == "identify_track":
            filepath = args.get("filepath")
            err = _validate_local_audio_path(filepath)
            if err:
                return {"ok": False, "error": err}
            info = identify_track(filepath)
            return {"ok": True, **info}

        if name == "get_music_trends":
            limit = int(args.get("limit", 10))
            country = args.get("country")
            chart = args.get("chart")
            genre = args.get("genre")
            save = bool(args.get("save", False))
            result = get_music_trends(limit=limit, country=country, chart=chart, genre=genre)
            items = result.get("items", [])
            item_type = result.get("item_type", "tracks")
            saved = 0
            saved_tracks = 0
            if save and item_type == "tracks":
                saved = db_store_external_tracks(
                    db,
                    items,
                    source="deezer",
                    chart=chart or item_type,
                    country=country,
                    genre=genre,
                )
                saved_tracks = db_store_external_as_tracks(
                    db,
                    items,
                    source="deezer",
                    genre=genre,
                )
            return {
                "ok": True,
                "country": result.get("country", country),
                "genre": result.get("genre", genre),
                "chart": chart or item_type,
                "item_type": item_type,
                "items": items,
                "saved": saved,
                "saved_tracks": saved_tracks if save else 0,
                "source": "Deezer",
            }

        if name == "detect_genre":
            filepath = args.get("filepath")
            err = _validate_local_audio_path(filepath)
            if err:
                return {"ok": False, "error": err}
            k = int(args.get("k", 5))
            result = predict_genre_for_file(db, filepath, k=k)
            return {"ok": True, **result}

        if name == "tag_audio":
            filepath = args.get("filepath")
            err = _validate_local_audio_path(filepath)
            if err:
                return {"ok": False, "error": err}
            top_k = int(args.get("top_k", 5))
            result = tag_audio(filepath, top_k=top_k)
            return {"ok": True, **result}

        if name == "text_to_speech":
            text = args.get("text")
            if not text:
                return {"ok": False, "error": "Missing text"}
            filename = args.get("filename")
            rate = args.get("rate")
            voice = args.get("voice")
            result = text_to_speech(
                text,
                output_dir=str(TTS_DIR),
                filename=filename,
                rate=rate,
                voice=voice,
            )
            tts_url = None
            try:
                out_path = Path(result["output_path"]).resolve()
                tts_root = TTS_DIR.resolve()
                rel = out_path.relative_to(tts_root)
                if len(rel.parts) == 1:
                    tts_url = f"/tts/{rel.name}"
            except ValueError:
                tts_url = None
            return {"ok": True, "output_path": result["output_path"], "tts_url": tts_url}

        if name == "tag_track":
            track_id = args.get("track_id")
            filepath = args.get("filepath")
            top_k = int(args.get("top_k", 5))

            if track_id:
                track = db_get_track(db, int(track_id))
                if not track:
                    return {"ok": False, "error": "Track not found"}
                filepath = track.get("filepath")
                if not filepath:
                    return {"ok": False, "error": "Track has no filepath"}

            if not filepath:
                return {"ok": False, "error": "Missing track_id or filepath"}

            result = tag_audio(filepath, top_k=top_k)
            if track_id:
                db_update_embedding_metadata(
                    db,
                    int(track_id),
                    {"ai_tags": result.get("tags", []), "ai_tag_model": result.get("model")},
                )
            return {
                "ok": True,
                "track_id": int(track_id) if track_id else None,
                "filepath": filepath,
                **result,
            }

        if name == "create_playlist":
            name = args.get("name")
            if not name:
                return {"ok": False, "error": "Missing name"}
            desc = args.get("description")
            pid = db_create_playlist(db, name, desc)
            return {"ok": True, "playlist_id": pid, "name": name, "description": desc}

        if name == "list_playlists":
            return {"ok": True, "playlists": db_list_playlists(db)}

        if name == "get_playlist":
            pid = args.get("playlist_id")
            if not pid:
                return {"ok": False, "error": "Missing playlist_id"}
            data = db_get_playlist(db, int(pid))
            return {"ok": True, **data}

        if name == "add_to_playlist":
            pid = args.get("playlist_id")
            tid = args.get("track_id")
            if not pid or not tid:
                return {"ok": False, "error": "Missing playlist_id or track_id"}
            pos = db_add_track_to_playlist(db, int(pid), int(tid))
            return {"ok": True, "playlist_id": int(pid), "track_id": int(tid), "position": pos}

        if name == "remove_from_playlist":
            pid = args.get("playlist_id")
            pos = args.get("position")
            if not pid or pos is None:
                return {"ok": False, "error": "Missing playlist_id or position"}
            db_remove_track_from_playlist(db, int(pid), int(pos))
            return {"ok": True, "playlist_id": int(pid), "position": int(pos)}

        if name == "delete_playlist":
            pid = args.get("playlist_id")
            if not pid:
                return {"ok": False, "error": "Missing playlist_id"}
            db_delete_playlist(db, int(pid))
            return {"ok": True, "playlist_id": int(pid)}

        return {"ok": False, "error": f"Unknown tool: {name}"}

    except Exception as e:
        return {"ok": False, "error": str(e)}


# =============================================================================
# AGENT
# =============================================================================

class Agent:
    SYSTEM = (
        "You are a music analysis agent. Your job is to answer user questions by planning and "
        "calling tools as needed. Use tools to read or compute data; do not guess results. "
        "When a question needs multiple steps, call multiple tools and combine the results "
        "into a clear, structured answer. Prefer short tool chains over long ones, but "
        "always use the tools when they exist.\n\n"
        "Tooling principles:\n"
        "- Plan first: identify the data needed and the tools that can fetch or compute it.\n"
        "- Decide if the user is asking you to do something; if yes, select the most appropriate tool(s).\n"
        "- Use database tools for track metadata and embeddings.\n"
        "- Use analysis tools for similarity, genre prediction, and statistics.\n"
        "- Use visualization/report tools when the user asks for reports or plots.\n"
        "- Use external API tools only when the task requires external data.\n"
        "- If a tool fails, explain the error and suggest the next best tool or input.\n\n"
        "Multi-step examples:\n"
        "- 'Analyze the library and generate a report' -> analyze_library, then generate_report.\n"
        "- 'Find similar tracks to 12 and make a workout playlist' -> recommend_tracks, then generate_playlist.\n"
        "- 'Download from YouTube and analyze' -> download_youtube_audio, then analyze_library/generate_report.\n"
        "- 'Detect genre and tag this file' -> detect_genre, then tag_audio.\n\n"
        "Output style:\n"
        "- Be concise, factual, and structured.\n"
        "- If tool output includes file paths or URLs, surface them clearly.\n"
        "- If the user asks for a download, provide the link when available.\n"
        "- Do not suggest Spotify or external playlists unless a tool explicitly returns those links."
    )

    def __init__(self, llm: Optional[LLM] = None, db: Optional[str] = None):
        self.llm = llm or get_llm()
        self.db = db or str(DB_PATH)

    def _extract_filepath(self, text: str) -> Optional[str]:
        patterns = [
            r"([A-Za-z]:\\[^\\n\"]+\\.(?:wav|mp3|flac|ogg|au|m4a|aac|wma))",
            r"(/[^\\s\"]+\\.(?:wav|mp3|flac|ogg|au|m4a|aac|wma))",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _extract_format(self, text: str) -> Optional[str]:
        formats = ["wav", "mp3", "flac", "ogg", "au", "m4a", "aac", "wma"]
        t = text.lower()
        
        # Pattern: "to mp3", "into wav", "as flac"
        pattern = r"\b(?:to|into|as|in)\s+\.?(" + "|".join(formats) + r")\b"
        match = re.search(pattern, t)
        if match:
            return match.group(1)
        
        # Pattern: "convert to .mp3", "make it .wav"
        pattern2 = r"\.(" + "|".join(formats) + r")\b"
        match = re.search(pattern2, t)
        if match:
            return match.group(1)
        
        # Pattern: "mp3 format", "wav file"
        pattern3 = r"\b(" + "|".join(formats) + r")\s*(?:format|file)?\b"
        match = re.search(pattern3, t)
        if match:
            return match.group(1)
        
        return None

    def _extract_tts_text(self, text: str) -> Optional[str]:
        lower = text.lower()
        for key in ["say:", "tts:", "speak:", "read:"]:
            if key in lower:
                return text.split(":", 1)[1].strip()
        for key in ["text to speech", "tts"]:
            if key in lower:
                return text.split(key, 1)[1].strip()
        return None

    def _parse_label(self, text: str) -> Optional[str]:
        if not text:
            return None
        match = re.search(r"(DATA_REQUIRED|META|CLARIFY|GENERAL)", text.upper())
        return match.group(1) if match else None

    def _parse_tool_toggle(self, text: str) -> Optional[bool]:
        if not text:
            return None
        match = re.search(r"(TOOL_REQUIRED|NO_TOOL)", text.upper())
        if not match:
            return None
        return match.group(1) == "TOOL_REQUIRED"

    def _classify_intent(self, text: str, has_file: bool) -> Optional[str]:
        if not self.llm:
            return None
        system = (
            "Classify the user message into exactly one label:\n"
            "DATA_REQUIRED: needs project data, files, database, reports, or external APIs.\n"
            "META: questions about the project, tools, setup, or capabilities.\n"
            "CLARIFY: missing info required to answer.\n"
            "GENERAL: general music theory or advice not tied to project data.\n"
            "Return only the label."
        )
        user = f"User message: {text}\nFile attached: {'yes' if has_file else 'no'}"
        resp = self.llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
        label = self._parse_label(resp.get("content", ""))
        if label:
            return label
        system_retry = "Return only one label: DATA_REQUIRED, META, CLARIFY, GENERAL."
        resp = self.llm.chat([{"role": "system", "content": system_retry}, {"role": "user", "content": user}])
        return self._parse_label(resp.get("content", ""))

    def _requires_tooling(self, text: str, has_file: bool) -> Optional[bool]:
        if not self.llm:
            return None
        system = (
            "Decide if this request requires calling tools. Return only TOOL_REQUIRED or NO_TOOL.\n"
            "TOOL_REQUIRED: any action that needs project data, analysis, reports, conversions, "
            "downloads, identification, tagging, trends, or file/database access.\n"
            "NO_TOOL: general explanations, definitions, or meta discussion without data access.\n"
            "If unsure, choose TOOL_REQUIRED."
        )
        user = f"User message: {text}\nFile attached: {'yes' if has_file else 'no'}"
        resp = self.llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
        toggle = self._parse_tool_toggle(resp.get("content", ""))
        if toggle is not None:
            return toggle
        system_retry = "Return only TOOL_REQUIRED or NO_TOOL."
        resp = self.llm.chat([{"role": "system", "content": system_retry}, {"role": "user", "content": user}])
        return self._parse_tool_toggle(resp.get("content", ""))

    def _extract_tool_plan(self, text: str) -> Optional[List[Dict[str, Any]]]:
        if not text:
            return None
        cleaned = text.strip()
        if "```" in cleaned:
            parts = cleaned.split("```")
            for part in parts:
                if "json" in part.lower():
                    cleaned = part.split("\n", 1)[-1].strip()
                    break
        if "{" not in cleaned or "}" not in cleaned:
            return None
        payload = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return None
        tool_calls = data.get("tool_calls") or data.get("tools")
        if isinstance(tool_calls, list):
            return tool_calls
        return None

    def _force_tool_plan(self, text: str, filepath: Optional[str]) -> Optional[List[Dict[str, Any]]]:
        if not self.llm:
            return None
        
        system = '''You are a tool selector. Return ONLY valid JSON, no prose.

AVAILABLE TOOLS:
- analyze_library: Get library statistics (no args needed)
- recommend_tracks: Find similar tracks (args: track_id, k)
- compare_tracks: Compare two tracks (args: track_a, track_b)
- find_by_mood: Find tracks by mood (args: mood like "happy", "sad", "energetic", "calm")
- generate_playlist: Generate playlist (args: theme like "workout", "relax", "study", "party")
- filter_by_tempo: Search by BPM (args: min_bpm, max_bpm)
- get_track_info: Get track info (args: track_id)
- generate_report: Generate report (args: scope "track" or "library", plus track_id or filepath for track scope)
- detect_genre: Predict genre (args: filepath)
- convert_audio: Convert format (args: filepath, output_format like "mp3", "wav", "flac")
- identify_track: Identify song (args: filepath)
- tag_audio: AI tagging (args: filepath)
- download_youtube_audio: Download from YouTube (args: url)
- get_music_trends: Get trending music from Deezer (args: country, chart, genre, limit, save)
- analyze_upload: Analyze uploaded file (args: filepath)
- save_upload: Save an uploaded file to the database (args: filepath)

EXAMPLES:
User: "analyze my library" -> {"tool_calls":[{"name":"analyze_library","arguments":{}}]}
User: "convert to mp3" + file -> {"tool_calls":[{"name":"convert_audio","arguments":{"output_format":"mp3"}}]}
User: "find similar to track 5" -> {"tool_calls":[{"name":"recommend_tracks","arguments":{"track_id":5}}]}
User: "compare 10 and 20" -> {"tool_calls":[{"name":"compare_tracks","arguments":{"track_a":10,"track_b":20}}]}
User: "playlist for workout" -> {"tool_calls":[{"name":"generate_playlist","arguments":{"theme":"workout"}}]}
User: "tempo between 120-140" -> {"tool_calls":[{"name":"filter_by_tempo","arguments":{"min_bpm":120,"max_bpm":140}}]}
User: "tempo between 120 and 140" -> {"tool_calls":[{"name":"filter_by_tempo","arguments":{"min_bpm":120,"max_bpm":140}}]}
User: "what genre is this" + file -> {"tool_calls":[{"name":"detect_genre","arguments":{}}]}
User: "generate report" + file -> {"tool_calls":[{"name":"generate_report","arguments":{"scope":"track"}}]}
User: "identify this song" + file -> {"tool_calls":[{"name":"identify_track","arguments":{}}]}
User: "save this upload" + file -> {"tool_calls":[{"name":"save_upload","arguments":{}}]}

Return ONLY the JSON object, nothing else.'''

        user = f"User request: {text}"
        if filepath:
            user += f"\nFile attached: {filepath}"
        
        resp = self.llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
        return self._extract_tool_plan(resp.get("content", ""))

    def _run_tools_from_llm(self, text: str, filepath: Optional[str]) -> str:
        if not self.llm:
            return self._help()
        system = (
            self.SYSTEM
            + "\n\nThis request requires tool use. You must call one or more tools."
            + "\nIf the user asks about tempo/BPM ranges (e.g., 'tempo between 120 and 140'), call filter_by_tempo."
            + "\nYou must choose from the provided tool list only. Do not answer without tool calls."
        )
        user = text
        if filepath:
            user += f"\nAttached file path: {filepath}"
        tool_choice = "required" if self.llm.supports_tool_choice else None
        try:
            resp = self.llm.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tools=TOOLS,
                tool_choice=tool_choice,
            )
            tool_calls = resp.get("tool_calls") or []
        except Exception:
            tool_calls = []
        if not tool_calls:
            try:
                tool_calls = self._force_tool_plan(text, filepath) or []
            except Exception:
                tool_calls = []
        if tool_calls:
            lower_text = (text or "").lower()
            wants_library = any(
                k in lower_text
                for k in [
                    "library report",
                    "full report",
                    "complete report",
                    "rapport global",
                    "rapport complet",
                    "rapport de la bibliotheque",
                    "bibliotheque",
                ]
            )
            wants_library_report = (
                "library report" in lower_text
                or "full library" in lower_text
                or "whole library" in lower_text
                or ("library" in lower_text and "report" in lower_text)
            )
            wants_track_report = (
                "track report" in lower_text
                or "report on this file" in lower_text
                or "report on this track" in lower_text
                or ("report" in lower_text and ("this file" in lower_text or "uploaded" in lower_text))
            )
            outputs = []
            tool_names = []
            report_calls = []
            for call in tool_calls:
                tool_args = call.get("arguments") or {}
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {}
                tool_name = call["name"]
                tool_name = TOOL_ALIASES.get(tool_name, tool_name)
                tool_names.append(tool_name)
                if tool_name == "generate_report":
                    report_calls.append({"args": tool_args})
                if filepath and "filepath" not in tool_args:
                    if tool_name in {
                        "generate_report",
                        "detect_genre",
                        "tag_audio",
                        "identify_track",
                        "generate_waveform",
                        "convert_audio",
                        "save_upload",
                    }:
                        tool_args["filepath"] = filepath
                if tool_name == "convert_audio" and not tool_args.get("output_format"):
                    out_fmt = self._extract_format(text or "")
                    if out_fmt:
                        tool_args["output_format"] = out_fmt
                if tool_name == "generate_report":
                    has_track = bool(tool_args.get("track_id") or tool_args.get("filepath"))
                    if not has_track and wants_library:
                        tool_args["scope"] = "library"
                    elif not has_track and not tool_args.get("scope"):
                        return "Please provide a track ID or upload a file to generate a track report."
                r = run_tool(tool_name, tool_args, self.db, self.llm)
                outputs.append(self._fmt(tool_name, r))
            if wants_library_report:
                has_library = any(
                    (c.get("args") or {}).get("scope") == "library" for c in report_calls
                )
                if not has_library:
                    r = run_tool("generate_report", {"scope": "library"}, self.db, self.llm)
                    outputs.append(self._fmt("generate_report", r))
            if wants_track_report and filepath:
                has_track = any(
                    (c.get("args") or {}).get("scope") == "track"
                    or (c.get("args") or {}).get("filepath")
                    for c in report_calls
                )
                if not has_track:
                    r = run_tool("generate_report", {"scope": "track", "filepath": filepath}, self.db, self.llm)
                    outputs.append(self._fmt("generate_report", r))
            tool_result = "\n\n".join(outputs)
            if not self.llm:
                return tool_result
            try:
                system = (
                    "You are a music assistant. Provide a concise, helpful reply that "
                    "answers the user's request using the tool results. Do not call tools. "
                    "Keep links and file paths intact. Return the tool results verbatim first, "
                    "then add 2-3 short sentences of commentary that do not add new facts. "
                    "If the tool results include any URLs or file paths, add a short 'Links:' line "
                    "that repeats the most important ones exactly. "
                    "Do not invent links or add placeholder URLs (never use example.com). "
                    "If no links or paths are present, do not add a Links line. "
                    "Do not suggest Spotify or external playlists unless the tool results include them."
                )
                user_msg = (
                    f"User request: {text}\n\nTool results:\n{tool_result}\n\n"
                    "Write the final response."
                )
                resp = self.llm.chat(
                    [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
                )
                final = (resp.get("content") or "").strip()
                return final or tool_result
            except Exception:
                return tool_result
        
        # Heuristic fallback for file operations when LLM fails
        if filepath:
            t = (text or "").lower()
            if "convert" in t:
                out_fmt = self._extract_format(text or "")
                if out_fmt:
                    r = run_tool("convert_audio", {"filepath": filepath, "output_format": out_fmt}, self.db, self.llm)
                    return self._fmt("convert_audio", r)
            if any(w in t for w in ["save", "store", "keep"]):
                r = run_tool("save_upload", {"filepath": filepath}, self.db, self.llm)
                return self._fmt("save_upload", r)
            if "report" in t or "rapport" in t:
                r = run_tool("generate_report", {"filepath": filepath}, self.db, self.llm)
                return self._fmt("generate_report", r)
            if any(w in t for w in ["genre", "style", "detect"]):
                r = run_tool("detect_genre", {"filepath": filepath}, self.db, self.llm)
                return self._fmt("detect_genre", r)
            if "identify" in t or "shazam" in t:
                r = run_tool("identify_track", {"filepath": filepath}, self.db, self.llm)
                return self._fmt("identify_track", r)
            if "tag" in t:
                r = run_tool("tag_audio", {"filepath": filepath}, self.db, self.llm)
                return self._fmt("tag_audio", r)
            # Default: analyze the file
            r = run_tool("analyze_upload", {"filepath": filepath}, self.db, self.llm)
            return self._fmt("analyze_upload", r)
        
        return "Error: tool selection failed. Please rephrase your request."

    def _fmt(self, action: str, r: Dict) -> str:
        action = TOOL_ALIASES.get(action, action)
        if not r.get("ok"):
            return f"Error: {r.get('error')}"

        if action == "analyze_upload" and "tracks" in r:
            tracks = r.get("tracks", [])
            if tracks:
                top = tracks[0]
                genre = top.get("genre") or top.get("true_genre")
                sim = top.get("sim")
                header = f"Likely genre: {genre}" if genre else "Likely genre: Unknown"
                if sim is not None:
                    header += f" (top match {sim})"
                lines = [header, "", "**Results**"]
                for i, t in enumerate(tracks, 1):
                    title = t.get("title") or f"Track {t.get('id')}"
                    genre = t.get("genre") or t.get("true_genre")
                    parts = []
                    if genre:
                        parts.append(genre)
                    if "sim" in t:
                        parts.append(f"sim {t['sim']}")
                    if "tempo_bpm" in t:
                        parts.append(f"{t['tempo_bpm']} bpm")
                    if "energy" in t:
                        parts.append(f"energy {t['energy']}")
                    suffix = f" ({', '.join(parts)})" if parts else ""
                    lines.append(f"{i}. {title}{suffix}")
                return "\n".join(lines)

        if "tracks" in r:
            lines = ["**Results**"]
            for i, t in enumerate(r.get("tracks", []), 1):
                title = t.get("title") or f"Track {t.get('id')}"
                genre = t.get("genre") or t.get("true_genre")
                parts = []
                if genre:
                    parts.append(genre)
                if "sim" in t:
                    parts.append(f"sim {t['sim']}")
                if "tempo_bpm" in t:
                    parts.append(f"{t['tempo_bpm']} bpm")
                if "energy" in t:
                    parts.append(f"energy {t['energy']}")
                suffix = f" ({', '.join(parts)})" if parts else ""
                lines.append(f"{i}. {title}{suffix}")
            return "\n".join(lines)

        if action == "generate_report":
            path = r.get("report_path")
            scope = r.get("scope") or "track"
            prefix = "Library report" if scope == "library" else "Report"
            lines = [f"{prefix} saved to {path}"]
            if r.get("html_url"):
                lines.append(f"Preview: {r['html_url']}")
            if r.get("report_url"):
                lines.append(f"Download (Markdown): {r['report_url']}")
            if r.get("pdf_url"):
                lines.append(f"Download (PDF): {r['pdf_url']}")
            plots = ", ".join(r.get("plots", []))
            if plots:
                lines.append(f"Plots: {plots}")
            return "\n".join(lines)

        if action == "create_playlist":
            return f"Playlist created: {r.get('name')} (id {r.get('playlist_id')})"

        if action == "list_playlists":
            pls = r.get("playlists", [])
            if not pls:
                return "No playlists found."
            lines = ["Playlists:"]
            for p in pls:
                desc = f" - {p.get('description')}" if p.get("description") else ""
                lines.append(f"- {p.get('id')}: {p.get('name')}{desc}")
            return "\n".join(lines)

        if action == "get_playlist":
            pl = r.get("playlist", {})
            tracks = r.get("tracks", [])
            lines = [f"Playlist {pl.get('id')}: {pl.get('name')}"]
            if pl.get("description"):
                lines.append(f"Description: {pl.get('description')}")
            if not tracks:
                lines.append("No tracks.")
                return "\n".join(lines)
            lines.append("")
            lines.append("Tracks:")
            for t in tracks:
                title = t.get("title") or f"Track {t.get('track_id')}"
                artist = t.get("artist") or "Unknown"
                lines.append(f"{t.get('position')}. {title} - {artist}")
            return "\n".join(lines)

        if action == "add_to_playlist":
            return (
                f"Added track {r.get('track_id')} to playlist {r.get('playlist_id')} "
                f"at position {r.get('position')}"
            )

        if action == "remove_from_playlist":
            return (
                f"Removed position {r.get('position')} from playlist {r.get('playlist_id')}"
            )

        if action == "delete_playlist":
            return f"Deleted playlist {r.get('playlist_id')}"

        if action == "generate_waveform":
            path = r.get("image_path")
            url = r.get("image_url")
            if url:
                return f"Waveform image: {url}"
            return f"Waveform image saved to {path}" if path else "Waveform image generated"

        if action == "save_upload":
            title = r.get("title") or "Unknown"
            artist = r.get("artist") or "Unknown"
            genre = r.get("genre") or "unknown"
            track_id = r.get("track_id")
            filepath = r.get("filepath")
            lines = [
                f"Saved track {track_id}: {title} - {artist}",
                f"Genre: {genre}",
            ]
            if filepath:
                lines.append(f"File: {filepath}")
            return "\n".join(lines)

        if action == "convert_audio":
            out_path = r.get("output_path")
            dl = r.get("download_url")
            if dl:
                return f"Conversion complete.\nFile: {out_path}\nDownload: {dl}"
            return f"Conversion complete.\nFile: {out_path}" if out_path else json.dumps(r, indent=2)

        if action == "identify_track":
            title = r.get("title")
            artist = r.get("artist")
            parts = []
            if title:
                parts.append(title)
            if artist:
                parts.append(artist)
            header = " - ".join(parts) if parts else "No match"
            lines = [header]
            if r.get("genres"):
                lines.append(f"Genre: {r['genres']}")
            if r.get("album"):
                lines.append(f"Album: {r['album']}")
            if r.get("shazam_url"):
                lines.append(f"More: {r['shazam_url']}")
            return "\n".join(lines)

        if action == "get_music_trends":
            items = r.get("items") or r.get("tracks", [])
            country = r.get("country")
            genre = r.get("genre")
            chart = r.get("chart")
            item_type = r.get("item_type") or "tracks"
            header_bits = []
            if chart:
                header_bits.append(chart)
            if genre:
                header_bits.append(genre)
            header = "Top trends"
            if header_bits:
                header += " (" + ", ".join(header_bits) + ")"
            elif country:
                header += f" ({country})"
            lines = [header, "", "**Results**"]
            for i, item in enumerate(items, 1):
                url = item.get("url")
                if item_type == "tracks":
                    title = item.get("title") or item.get("name") or "Unknown"
                    artist = item.get("artist") or "Unknown"
                    line = f"{i}. {title} - {artist}"
                elif item_type == "albums":
                    title = item.get("title") or "Unknown"
                    artist = item.get("artist") or "Unknown"
                    line = f"{i}. {title} - {artist}"
                elif item_type == "artists":
                    name = item.get("name") or "Unknown"
                    line = f"{i}. {name}"
                elif item_type == "playlists":
                    title = item.get("title") or "Unknown"
                    creator = item.get("creator") or "Unknown"
                    line = f"{i}. {title} - {creator}"
                else:
                    line = f"{i}. {item.get('title') or item.get('name') or 'Unknown'}"
                if url:
                    line += f" ({url})"
                lines.append(line)
            saved = r.get("saved", 0)
            saved_tracks = r.get("saved_tracks", 0)
            if saved or saved_tracks:
                lines.append("")
                if saved:
                    lines.append(f"Saved to database (external): {saved} track(s)")
                if saved_tracks:
                    lines.append(f"Saved to library: {saved_tracks} track(s)")
            return "\n".join(lines)

        if action == "detect_genre":
            genre = r.get("predicted_genre") or "Unknown"
            confidence = r.get("confidence", 0.0)
            reason = r.get("reason") or "No explanation available."
            features = r.get("features", {})
            matches = r.get("top_matches", [])[:3]

            lines = [f"Predicted genre: {genre} (confidence {confidence:.2f})"]
            lines.append(f"Why: {reason}")
            if features:
                lines.append(
                    "Audio profile: "
                    f"{features.get('tempo_bpm', 0):.1f} BPM, "
                    f"energy {features.get('energy', 0):.2f}, "
                    f"spectral centroid {features.get('spectral_centroid', 0):.0f}, "
                    f"valence {features.get('valence', 0):.2f}"
                )
            if matches:
                lines.append("Closest references:")
                for m in matches:
                    title = m.get("title") or f"Track {m.get('id')}"
                    lines.append(
                        f"- {title} ({m.get('genre')}, sim {m.get('sim', 0):.3f})"
                    )
            return "\n".join(lines)

        if action == "tag_audio":
            tags = r.get("tags", [])
            model = r.get("model") or "model"
            lines = [f"Top tags ({model}):"]
            for t in tags:
                lines.append(f"- {t.get('tag')}: {t.get('score', 0):.2f}")
            return "\n".join(lines)

        if action == "tag_track":
            tags = r.get("tags", [])
            model = r.get("model") or "model"
            track_id = r.get("track_id")
            header = f"Tags saved for track {track_id}" if track_id else "Tags for file"
            lines = [header, f"Model: {model}"]
            for t in tags:
                lines.append(f"- {t.get('tag')}: {t.get('score', 0):.2f}")
            return "\n".join(lines)

        if action == "text_to_speech":
            out_path = r.get("output_path")
            url = r.get("tts_url")
            if url:
                return f"TTS audio saved to {out_path}\nDownload: {url}"
            return f"TTS audio saved to {out_path}" if out_path else json.dumps(r, indent=2)

        if action == "analyze_library":
            total = r.get("total", 0)
            coverage = r.get("coverage", {})
            tempo = r.get("tempo_stats", {})
            energy = r.get("energy_stats", {})
            top_genres = r.get("top_genres", [])[:5]
            outliers = r.get("outliers", {})

            lines = [f"Library summary: {total} tracks"]
            if coverage:
                lines.append(
                    f"Embeddings: {coverage.get('tracks_with_embeddings', 0)} ok, "
                    f"{coverage.get('tracks_missing_embeddings', 0)} missing"
                )
            if top_genres:
                tg = ", ".join(
                    f"{g['genre']} ({g['count']}, {g['pct']:.1f}%)" for g in top_genres
                )
                lines.append(f"Top genres: {tg}")
            if tempo:
                lines.append(
                    "Tempo BPM: "
                    f"mean {tempo.get('mean', 0):.1f}, median {tempo.get('median', 0):.1f}, "
                    f"p25 {tempo.get('p25', 0):.1f}, p75 {tempo.get('p75', 0):.1f}, "
                    f"min {tempo.get('min', 0):.1f}, max {tempo.get('max', 0):.1f}"
                )
            if energy:
                lines.append(
                    "Energy: "
                    f"mean {energy.get('mean', 0):.2f}, median {energy.get('median', 0):.2f}, "
                    f"p25 {energy.get('p25', 0):.2f}, p75 {energy.get('p75', 0):.2f}, "
                    f"min {energy.get('min', 0):.2f}, max {energy.get('max', 0):.2f}"
                )
            fastest = outliers.get("fastest", {})
            slowest = outliers.get("slowest", {})
            energetic = outliers.get("most_energetic", {})
            if fastest:
                lines.append(
                    f"Fastest: {fastest.get('title') or fastest.get('id')} "
                    f"({fastest.get('genre')}, {fastest.get('tempo_bpm', 0):.1f} BPM)"
                )
            if slowest:
                lines.append(
                    f"Slowest: {slowest.get('title') or slowest.get('id')} "
                    f"({slowest.get('genre')}, {slowest.get('tempo_bpm', 0):.1f} BPM)"
                )
            if energetic:
                lines.append(
                    f"Most energetic: {energetic.get('title') or energetic.get('id')} "
                    f"({energetic.get('genre')}, {energetic.get('energy', 0):.2f})"
                )
            return "\n".join(lines)

        if action == "download_youtube_audio":
            title = r.get("title") or "Unknown"
            artist = r.get("artist") or "Unknown"
            filepath = r.get("filepath")
            dl = r.get("download_url")
            lines = [f"YouTube download complete: {title} - {artist}"]
            if filepath:
                lines.append(f"File: {filepath}")
            if dl:
                lines.append(f"Download: {dl}")
            return "\n".join(lines)

        return json.dumps(r, indent=2)

    def _help(self) -> str:
        return "\n".join(
            [
                "Try:",
                "- Analyze the library",
                "- Recommend 5 tracks similar to track 30",
                "- Compare track 12 and 47",
                "- Create a workout playlist",
                "- Search tracks between 100 and 130 BPM",
                "- Generate a report (track or library)",
                "- Convert a file to mp3",
                "- Identify a track from a file",
                "- Show music trends",
                "- Detect genre from a file",
                "- Tag audio from a file",
                "- Save an uploaded file to the database",
                "- Convert text to speech",
                "- Tag an existing track",
                "- Create a playlist",
                "- Add track 12 to playlist 3",
            ]
        )

    def _parse(self, text: str) -> Dict:
        t = text.lower()
        nums = [int(n) for n in re.findall(r"\d+", t)]
        urls = re.findall(r"https?://\S+", text)
        filepath = self._extract_filepath(text)
        out_fmt = self._extract_format(text)
        tts_text = self._extract_tts_text(text)

        if "convert" in t and filepath and out_fmt:
            return {
                "action": "convert_audio",
                "args": {"filepath": filepath, "output_format": out_fmt},
            }
        if ("report" in t or "rapport" in t) and filepath:
            return {"action": "generate_report", "args": {"filepath": filepath}}
        if ("identify" in t or "shazam" in t) and filepath:
            return {"action": "identify_track", "args": {"filepath": filepath}}
        if "waveform" in t or "preview" in t:
            if filepath:
                return {"action": "generate_waveform", "args": {"filepath": filepath}}
        if any(w in t for w in ["genre", "style", "detect genre"]) and filepath:
            return {"action": "detect_genre", "args": {"filepath": filepath}}
        if "tag" in t and filepath:
            return {"action": "tag_audio", "args": {"filepath": filepath}}
        if any(w in t for w in ["text to speech", "tts"]) and tts_text:
            return {"action": "text_to_speech", "args": {"text": tts_text}}

        if urls and ("youtube" in t or "youtu.be" in t or "download" in t):
            return {"action": "download_youtube_audio", "args": {"url": urls[0]}}
        if any(w in t for w in [" and ", " then ", ","]):
            actions = []
            if "library report" in t or "rapport global" in t:
                actions.append({"action": "generate_report", "args": {"scope": "library"}})
            if "report" in t or "rapport" in t:
                actions.append({"action": "generate_report", "args": {}})
            if "analyse" in t or "analyze" in t:
                actions.append({"action": "analyze_library", "args": {}})
            if "trend" in t:
                args = {}
                if any(w in t for w in ["save", "store", "persist"]):
                    args["save"] = True
                actions.append({"action": "get_music_trends", "args": args})
            if "tts" in t or "text to speech" in t:
                if tts_text:
                    actions.append({"action": "text_to_speech", "args": {"text": tts_text}})
            if "playlist" in t:
                theme = t.split("playlist", 1)[1].strip() or "mix"
                actions.append({"action": "generate_playlist", "args": {"theme": theme, "k": 8}})
            if "tempo" in t or "bpm" in t:
                if len(nums) >= 2:
                    actions.append({"action": "filter_by_tempo", "args": {"min_bpm": nums[0], "max_bpm": nums[1]}})
            if any(w in t for w in ["similar", "similaire", "recommend", "recommande"]):
                actions.append({"action": "recommend_tracks", "args": {"track_id": nums[0] if nums else 1, "k": 5}})
            if "compare" in t and len(nums) >= 2:
                actions.append({"action": "compare_tracks", "args": {"track_a": nums[0], "track_b": nums[1]}})
            if actions:
                return {"actions": actions}
        if any(
            k in t
            for k in [
                "library report",
                "full report",
                "complete report",
                "rapport global",
                "rapport complet",
                "rapport de la bibliotheque",
                "bibliotheque",
            ]
        ):
            return {"action": "generate_report", "args": {"scope": "library"}}
        if "report" in t or "rapport" in t:
            if nums:
                return {"action": "generate_report", "args": {"track_id": nums[0]}}
            return {"action": "generate_report", "args": {"scope": "library"}}
        if "create playlist" in t:
            name = text.split("playlist", 1)[1].strip() or "My Playlist"
            return {"action": "create_playlist", "args": {"name": name}}
        if "list playlists" in t:
            return {"action": "list_playlists", "args": {}}
        if "playlist" in t and "show" in t and nums:
            return {"action": "get_playlist", "args": {"playlist_id": nums[0]}}
        if "add" in t and "playlist" in t and len(nums) >= 2:
            return {"action": "add_to_playlist", "args": {"playlist_id": nums[0], "track_id": nums[1]}}
        if "remove" in t and "playlist" in t and len(nums) >= 2:
            return {"action": "remove_from_playlist", "args": {"playlist_id": nums[0], "position": nums[1]}}
        if "delete playlist" in t and nums:
            return {"action": "delete_playlist", "args": {"playlist_id": nums[0]}}
        if "analyse" in t or "analyze" in t:
            return {"action": "analyze_library", "args": {}}
        if "compare" in t and len(nums) >= 2:
            return {"action": "compare_tracks", "args": {"track_a": nums[0], "track_b": nums[1]}}
        if any(w in t for w in ["similar", "similaire", "recommend", "recommande"]):
            return {"action": "recommend_tracks", "args": {"track_id": nums[0] if nums else 1, "k": 5}}
        if "playlist" in t:
            theme = t.split("playlist", 1)[1].strip() or "mix"
            return {"action": "generate_playlist", "args": {"theme": theme, "k": 8}}
        if "tempo" in t or "bpm" in t:
            if len(nums) >= 2:
                return {"action": "filter_by_tempo", "args": {"min_bpm": nums[0], "max_bpm": nums[1]}}
        if "track" in t or "morceau" in t:
            if nums:
                return {"action": "get_track_info", "args": {"track_id": nums[0]}}
        mood_map = {
            "relax": "relaxed",
            "calm": "calm",
            "chill": "relaxed",
            "happy": "happy",
            "sad": "sad",
            "energetic": "energetic",
            "focus": "focused",
        }
        for key, mood in mood_map.items():
            if key in t:
                return {"action": "find_by_mood", "args": {"mood": mood, "k": 5}}
        return {"action": None, "args": {}}

    def chat(self, text: str, filepath: Optional[str] = None) -> str:
        if self.llm:
            try:
                label = self._classify_intent(text, filepath is not None)
                tool_required = self._requires_tooling(text, filepath is not None)
                t = (text or "").lower()
                meta_like = any(
                    k in t
                    for k in [
                        "what is",
                        "how does",
                        "explain",
                        "help",
                        "documentation",
                        "capability",
                        "tools",
                        "tool",
                    ]
                )
                needs_data = bool(filepath) or (
                    not meta_like
                    and any(
                        k in t
                        for k in [
                            "report",
                            "analyse",
                            "analyze",
                            "track",
                            "playlist",
                            "similar",
                            "compare",
                            "bpm",
                            "tempo",
                            "trend",
                            "tag",
                            "genre",
                            "convert",
                            "identify",
                            "shazam",
                            "youtube",
                            "download",
                        ]
                    )
                )
                if label != "DATA_REQUIRED" and needs_data:
                    system = "Re-evaluate. This request depends on project data. Return only DATA_REQUIRED."
                    resp = self.llm.chat(
                        [{"role": "system", "content": system}, {"role": "user", "content": text}]
                    )
                    label = self._parse_label(resp.get("content", "")) or "DATA_REQUIRED"

                if tool_required is None:
                    tool_required = label == "DATA_REQUIRED" or needs_data

                if tool_required or label == "DATA_REQUIRED":
                    return self._run_tools_from_llm(text, filepath)
                if label == "CLARIFY":
                    system = "Ask a short clarification question needed to answer. Do not use tools."
                    resp = self.llm.chat(
                        [{"role": "system", "content": system}, {"role": "user", "content": text}]
                    )
                    return resp.get("content") or "Could you clarify? Please provide more details."

                resp = self.llm.chat(
                    [{"role": "system", "content": self.SYSTEM}, {"role": "user", "content": text}]
                )
                return resp.get("content") or self._help()
            except Exception as e:
                return f"Error: {e}"

        if filepath:
            t = (text or "").lower()
            if any(w in t for w in ["save", "store", "keep"]):
                r = run_tool("save_upload", {"filepath": filepath}, self.db, self.llm)
                return self._fmt("save_upload", r)
            if "report" in t or "rapport" in t:
                r = run_tool("generate_report", {"filepath": filepath}, self.db, self.llm)
                return self._fmt("generate_report", r)
            if any(w in t for w in ["genre", "style", "genre detect", "detect genre"]):
                r = run_tool("detect_genre", {"filepath": filepath}, self.db, self.llm)
                return self._fmt("detect_genre", r)
            if "tag" in t:
                r = run_tool("tag_audio", {"filepath": filepath}, self.db, self.llm)
                return self._fmt("tag_audio", r)
            if "identify" in t or "shazam" in t:
                r = run_tool("identify_track", {"filepath": filepath}, self.db, self.llm)
                return self._fmt("identify_track", r)
            if "waveform" in t or "preview" in t:
                r = run_tool("generate_waveform", {"filepath": filepath}, self.db, self.llm)
                return self._fmt("generate_waveform", r)
            if "convert" in t:
                out_fmt = self._extract_format(text or "")
                if out_fmt:
                    r = run_tool(
                        "convert_audio",
                        {"filepath": filepath, "output_format": out_fmt},
                        self.db,
                    )
                    return self._fmt("convert_audio", r)
            r = run_tool("analyze_upload", {"filepath": filepath}, self.db, self.llm)
            return self._fmt("analyze_upload", r)

        parsed = self._parse(text)
        if parsed.get("actions"):
            outputs = []
            for item in parsed["actions"]:
                action = item.get("action")
                args = item.get("args", {})
                if action:
                    r = run_tool(action, args, self.db, self.llm)
                    outputs.append(self._fmt(action, r))
            return "\n\n".join(outputs)
        if parsed.get("action") == "download_youtube_audio":
            r = run_tool("download_youtube_audio", parsed.get("args", {}), self.db, self.llm)
            return self._fmt("download_youtube_audio", r)
        if parsed.get("action"):
            r = run_tool(parsed.get("action"), parsed.get("args", {}), self.db, self.llm)
            return self._fmt(parsed.get("action"), r)
        return self._help()
