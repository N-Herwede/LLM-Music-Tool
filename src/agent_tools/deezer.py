"""
Deezer trends helpers.
"""

from typing import Any, Dict, List, Optional
import json
import urllib.parse
import urllib.request

DEEZER_API_URL = "https://api.deezer.com"


def _fetch_json(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{DEEZER_API_URL}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "MusicAgentPro/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if isinstance(payload, dict) and payload.get("error"):
        msg = payload["error"].get("message") or "Deezer API error"
        raise RuntimeError(msg)
    return payload


def _match_editorial_id(country: Optional[str]) -> int:
    if not country:
        return 0
    target_raw = country.strip().lower()
    target = "".join(ch for ch in target_raw if ch.isalpha())
    data = _fetch_json("/editorial")
    for item in data.get("data", []):
        name = (item.get("name") or "").lower()
        code = (item.get("country") or "").lower()
        norm_name = "".join(ch for ch in name if ch.isalpha())
        norm_code = "".join(ch for ch in code if ch.isalpha())
        if target in (norm_name, norm_code):
            return int(item.get("id", 0))
    raise RuntimeError(f"Unknown country/editorial: {country}")


def _normalize_track(item: Dict[str, Any]) -> Dict[str, Any]:
    artist = item.get("artist") or {}
    album = item.get("album") or {}
    return {
        "id": item.get("id"),
        "title": item.get("title") or item.get("name"),
        "artist": artist.get("name"),
        "album": album.get("title"),
        "url": item.get("link"),
        "preview": item.get("preview"),
        "rank": item.get("rank"),
        "type": "track",
    }


def _normalize_album(item: Dict[str, Any]) -> Dict[str, Any]:
    artist = item.get("artist") or {}
    return {
        "id": item.get("id"),
        "title": item.get("title"),
        "artist": artist.get("name"),
        "url": item.get("link"),
        "rank": item.get("rank"),
        "type": "album",
    }


def _normalize_artist(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": item.get("id"),
        "name": item.get("name"),
        "url": item.get("link"),
        "rank": item.get("rank"),
        "type": "artist",
    }


def _normalize_playlist(item: Dict[str, Any]) -> Dict[str, Any]:
    user = item.get("user") or {}
    return {
        "id": item.get("id"),
        "title": item.get("title"),
        "creator": user.get("name"),
        "url": item.get("link"),
        "rank": item.get("rank"),
        "type": "playlist",
    }


def _get_genre_id(genre: str) -> int:
    data = _fetch_json("/genre")
    target = genre.strip().lower()
    for item in data.get("data", []):
        name = (item.get("name") or "").lower()
        if target == name:
            return int(item.get("id"))
    raise RuntimeError(f"Unknown genre: {genre}")


def _get_genre_tracks(genre_id: int, limit: int) -> List[Dict[str, Any]]:
    artists = _fetch_json(f"/genre/{genre_id}/artists").get("data", [])
    tracks: List[Dict[str, Any]] = []
    for artist in artists:
        artist_id = artist.get("id")
        if not artist_id:
            continue
        top = _fetch_json(f"/artist/{artist_id}/top", {"limit": 10}).get("data", [])
        for track in top:
            tracks.append(track)
            if len(tracks) >= limit:
                return tracks
    return tracks


def get_music_trends(
    limit: int = 10,
    country: Optional[str] = None,
    chart: Optional[str] = None,
    genre: Optional[str] = None,
) -> Dict[str, Any]:
    """Get Deezer trends by country, chart type, or genre."""
    item_type = (chart or "tracks").strip().lower()
    if genre:
        genre_id = _get_genre_id(genre)
        tracks = _get_genre_tracks(genre_id, limit)
        items = [_normalize_track(t) for t in tracks][:limit]
        return {
            "items": items,
            "item_type": "tracks",
            "country": country,
            "genre": genre,
        }

    editorial_id = _match_editorial_id(country)
    chart_data = _fetch_json(f"/editorial/{editorial_id}/charts")
    raw = chart_data.get(item_type, {}).get("data", [])

    if item_type == "tracks":
        items = [_normalize_track(t) for t in raw][:limit]
    elif item_type == "albums":
        items = [_normalize_album(t) for t in raw][:limit]
    elif item_type == "artists":
        items = [_normalize_artist(t) for t in raw][:limit]
    elif item_type == "playlists":
        items = [_normalize_playlist(t) for t in raw][:limit]
    else:
        raise RuntimeError(f"Unknown chart type: {item_type}")

    return {
        "items": items,
        "item_type": item_type,
        "country": country,
        "genre": None,
    }
