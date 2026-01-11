"""
Database helpers.
"""

import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@contextmanager
def get_db(db_path: str):
    """Database connection context manager."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_database(db_path: str):
    """Initialize database schema."""
    with get_db(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                artist TEXT,
                filepath TEXT UNIQUE,
                true_genre TEXT,
                predicted_genre TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                track_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                metadata TEXT,
                FOREIGN KEY (track_id) REFERENCES tracks(id)
            );

            CREATE INDEX IF NOT EXISTS idx_tracks_genre ON tracks(true_genre);

            CREATE TABLE IF NOT EXISTS playlists (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS playlist_tracks (
                playlist_id INTEGER NOT NULL,
                track_id INTEGER NOT NULL,
                position INTEGER NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (playlist_id, position),
                FOREIGN KEY (playlist_id) REFERENCES playlists(id),
                FOREIGN KEY (track_id) REFERENCES tracks(id)
            );

            CREATE INDEX IF NOT EXISTS idx_playlist_tracks_playlist ON playlist_tracks(playlist_id);

            CREATE TABLE IF NOT EXISTS external_tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                external_id TEXT NOT NULL,
                title TEXT,
                artist TEXT,
                album TEXT,
                url TEXT,
                preview_url TEXT,
                chart TEXT,
                country TEXT,
                genre TEXT,
                rank INTEGER,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source, external_id, chart, country, genre)
            );
            """
        )
        conn.commit()


def _ensure_external_tracks_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS external_tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            external_id TEXT NOT NULL,
            title TEXT,
            artist TEXT,
            album TEXT,
            url TEXT,
            preview_url TEXT,
            chart TEXT,
            country TEXT,
            genre TEXT,
            rank INTEGER,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source, external_id, chart, country, genre)
        )
        """
    )


def db_store_external_tracks(
    db_path: str,
    tracks: List[Dict[str, Any]],
    source: str,
    chart: Optional[str] = None,
    country: Optional[str] = None,
    genre: Optional[str] = None,
) -> int:
    """Store external chart tracks in the database."""
    inserted = 0
    with get_db(db_path) as conn:
        _ensure_external_tracks_table(conn)
        for item in tracks:
            ext_id = item.get("id")
            if ext_id is None:
                continue
            title = item.get("title")
            artist = item.get("artist")
            album = item.get("album")
            url = item.get("url")
            preview = item.get("preview")
            rank = item.get("rank")
            conn.execute(
                """
                INSERT OR REPLACE INTO external_tracks
                (source, external_id, title, artist, album, url, preview_url, chart, country, genre, rank)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source,
                    str(ext_id) if ext_id is not None else None,
                    title,
                    artist,
                    album,
                    url,
                    preview,
                    chart,
                    country,
                    genre,
                    int(rank) if rank is not None else None,
                ),
            )
            inserted += 1
        conn.commit()
    return inserted


def db_store_external_as_tracks(
    db_path: str,
    tracks: List[Dict[str, Any]],
    source: str,
    genre: Optional[str] = None,
) -> int:
    """Store external chart items into the main tracks table."""
    inserted = 0
    for item in tracks:
        ext_id = item.get("id")
        if ext_id is None:
            continue
        title = item.get("title") or item.get("name") or "Unknown"
        artist = item.get("artist") or "Unknown"
        item_genre = genre or "unknown"
        filepath = f"{source}:{ext_id}"
        track_id = db_add_track(db_path, title, filepath, item_genre, artist=artist)
        if track_id is not None:
            inserted += 1
    return inserted


def db_add_track(
    db_path: str,
    title: str,
    filepath: str,
    genre: str,
    artist: Optional[str] = None,
) -> Optional[int]:
    """Add a track to the database."""
    with get_db(db_path) as conn:
        existing = conn.execute(
            "SELECT id, title, artist, true_genre FROM tracks WHERE filepath = ?",
            (filepath,),
        ).fetchone()
        if existing:
            updates = {
                "title": title if title else existing["title"],
                "artist": artist if artist else existing["artist"],
                "true_genre": genre if genre else existing["true_genre"],
            }
            conn.execute(
                "UPDATE tracks SET title = ?, artist = ?, true_genre = ? WHERE filepath = ?",
                (updates["title"], updates["artist"], updates["true_genre"], filepath),
            )
            conn.commit()
            return int(existing["id"])

        cursor = conn.execute(
            "INSERT INTO tracks (title, artist, filepath, true_genre) VALUES (?, ?, ?, ?)",
            (title, artist, filepath, genre),
        )
        conn.commit()
        return cursor.lastrowid


def db_store_embedding(
    db_path: str,
    track_id: int,
    embedding: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Store embedding for a track."""
    with get_db(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (track_id, embedding, metadata) VALUES (?, ?, ?)",
            (track_id, embedding.tobytes(), json.dumps(metadata or {})),
        )
        conn.commit()


def db_update_embedding_metadata(
    db_path: str,
    track_id: int,
    updates: Dict[str, Any],
    replace: bool = False,
) -> None:
    """Update embedding metadata for a track."""
    with get_db(db_path) as conn:
        row = conn.execute(
            "SELECT metadata FROM embeddings WHERE track_id = ?",
            (track_id,),
        ).fetchone()
        if not row:
            raise ValueError("Embedding not found for track_id")
        current = json.loads(row[0]) if row[0] else {}
        merged = updates if replace else {**current, **updates}
        conn.execute(
            "UPDATE embeddings SET metadata = ? WHERE track_id = ?",
            (json.dumps(merged), track_id),
        )
        conn.commit()


def db_get_track(db_path: str, track_id: int) -> Optional[Dict]:
    """Get a track by ID."""
    with get_db(db_path) as conn:
        row = conn.execute("SELECT * FROM tracks WHERE id = ?", (track_id,)).fetchone()
        return dict(row) if row else None


def db_get_all_tracks(db_path: str) -> List[Dict]:
    """Get all tracks."""
    with get_db(db_path) as conn:
        rows = conn.execute("SELECT * FROM tracks ORDER BY id").fetchall()
        return [dict(r) for r in rows]


def db_get_all_embeddings(db_path: str) -> List[Tuple[int, bytes, Dict]]:
    """Get all embeddings with metadata."""
    with get_db(db_path) as conn:
        rows = conn.execute("SELECT track_id, embedding, metadata FROM embeddings").fetchall()
        return [(r[0], r[1], json.loads(r[2]) if r[2] else {}) for r in rows]


def db_create_playlist(db_path: str, name: str, description: Optional[str] = None) -> int:
    """Create a playlist and return its ID."""
    with get_db(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO playlists (name, description) VALUES (?, ?)",
            (name, description),
        )
        conn.commit()
        return int(cur.lastrowid)


def db_list_playlists(db_path: str) -> List[Dict]:
    """List all playlists."""
    with get_db(db_path) as conn:
        rows = conn.execute(
            "SELECT id, name, description, created_at FROM playlists ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]


def db_get_playlist(db_path: str, playlist_id: int) -> Dict[str, Any]:
    """Get playlist info and ordered tracks."""
    with get_db(db_path) as conn:
        pl = conn.execute(
            "SELECT id, name, description, created_at FROM playlists WHERE id = ?",
            (playlist_id,),
        ).fetchone()
        if not pl:
            raise ValueError("Playlist not found")
        rows = conn.execute(
            """
            SELECT pt.position, t.id AS track_id, t.title, t.artist, t.true_genre
            FROM playlist_tracks pt
            JOIN tracks t ON t.id = pt.track_id
            WHERE pt.playlist_id = ?
            ORDER BY pt.position
            """,
            (playlist_id,),
        ).fetchall()
        return {
            "playlist": dict(pl),
            "tracks": [dict(r) for r in rows],
        }


def db_add_track_to_playlist(db_path: str, playlist_id: int, track_id: int) -> int:
    """Append a track to a playlist and return its position."""
    with get_db(db_path) as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(position), 0) FROM playlist_tracks WHERE playlist_id = ?",
            (playlist_id,),
        ).fetchone()
        next_pos = int(row[0] or 0) + 1
        conn.execute(
            "INSERT INTO playlist_tracks (playlist_id, track_id, position) VALUES (?, ?, ?)",
            (playlist_id, track_id, next_pos),
        )
        conn.commit()
        return next_pos


def db_remove_track_from_playlist(db_path: str, playlist_id: int, position: int) -> None:
    """Remove a track from a playlist by position."""
    with get_db(db_path) as conn:
        conn.execute(
            "DELETE FROM playlist_tracks WHERE playlist_id = ? AND position = ?",
            (playlist_id, position),
        )
        conn.commit()


def db_delete_playlist(db_path: str, playlist_id: int) -> None:
    """Delete a playlist and its tracks."""
    with get_db(db_path) as conn:
        conn.execute("DELETE FROM playlist_tracks WHERE playlist_id = ?", (playlist_id,))
        conn.execute("DELETE FROM playlists WHERE id = ?", (playlist_id,))
        conn.commit()
