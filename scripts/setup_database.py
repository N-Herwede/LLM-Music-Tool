#!/usr/bin/env python3
"""
Database Setup Script
=====================

Initialize database and optionally ingest GTZAN dataset.

Usage:
    python scripts/setup_database.py              # Setup + ingest if GTZAN exists
    python scripts/setup_database.py --db-only    # Only create database schema
    python scripts/setup_database.py --force      # Reset everything
"""

import argparse
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_tools.database import init_database, db_add_track, db_store_embedding
from agent_tools.analysis import extract_features, build_feature_vector

# Paths
DB_PATH = ROOT / "data" / "processed" / "music_library.db"
GTZAN_PATH = ROOT / "data" / "raw" / "gtzan"

# GTZAN genres
GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]


def setup_database(force: bool = False) -> None:
    """Initialize the database."""
    print("Setting up database...")
    
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if force and DB_PATH.exists():
        print("   Removing existing database...")
        DB_PATH.unlink()
    
    init_database(str(DB_PATH))
    print(f"   Database ready: {DB_PATH}")


def find_audio_files() -> list:
    """Find all audio files in GTZAN directory."""
    files = []
    
    if not GTZAN_PATH.exists():
        return files
    
    # Look in genre subdirectories
    for genre in GENRES:
        genre_dir = GTZAN_PATH / genre
        if genre_dir.exists():
            for ext in ["*.wav", "*.au", "*.mp3"]:
                files.extend([(f, genre) for f in genre_dir.glob(ext)])
    
    # Also look for flat files
    for ext in ["*.wav", "*.au", "*.mp3"]:
        for f in GTZAN_PATH.glob(ext):
            # Try to extract genre from filename
            name = f.stem.lower()
            genre = next((g for g in GENRES if g in name), "unknown")
            files.append((f, genre))
    
    return files


def ingest_gtzan() -> int:
    """Ingest GTZAN dataset into the database."""
    print("\nLooking for GTZAN dataset...")
    
    files = find_audio_files()
    
    if not files:
        print(f"   No audio files found in {GTZAN_PATH}")
        return 0
    
    print(f"   Found {len(files)} audio files")
    print("\nIngesting tracks...")
    
    success = 0
    errors = 0
    
    for i, (filepath, genre) in enumerate(files, 1):
        try:
            # Extract features
            features = extract_features(str(filepath))
            embedding = build_feature_vector(str(filepath))
            
            # Add to database
            title = filepath.stem
            track_id = db_add_track(str(DB_PATH), title, str(filepath), genre)
            
            if track_id is None:
                raise ValueError("Failed to insert track")
            
            # Store embedding with metadata
            metadata = {
                "tempo_bpm": features.get("tempo_bpm", 0),
                "energy": features.get("energy", 0),
                "valence": features.get("valence", 0),
                "spectral_centroid": features.get("spectral_centroid", 0),
            }
            db_store_embedding(str(DB_PATH), track_id, embedding, metadata)
            
            success += 1
            
            # Progress update
            if i % 50 == 0 or i == len(files):
                print(f"   Progress: {i}/{len(files)} ({success} ok, {errors} fail)")
                
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"   Error with {filepath.name}: {e}")
    
    print(f"\n   Ingested {success} tracks successfully")
    if errors > 0:
        print(f"   Errors: {errors} tracks failed")
    
    return success


def main():
    parser = argparse.ArgumentParser(description="Setup Music Agent database")
    parser.add_argument("--db-only", action="store_true", help="Only create database schema")
    parser.add_argument("--force", action="store_true", help="Reset database if exists")
    args = parser.parse_args()
    
    print("Music Agent Pro - Database Setup")
    
    # Setup database
    setup_database(force=args.force)
    
    # Ingest data unless --db-only
    if not args.db_only:
        count = ingest_gtzan()
        
        if count == 0:
            print("\nTo add the GTZAN dataset:")
            print("   1. Download from: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification")
            print(f"   2. Extract to: {GTZAN_PATH}")
            print("   3. Run: python scripts/setup_database.py")
    
    print("Setup complete!")
    print("\nNext steps:")
    print("   python scripts/server.py    # Start web interface")
    print("   python scripts/cli.py       # Terminal interface")


if __name__ == "__main__":
    main()
