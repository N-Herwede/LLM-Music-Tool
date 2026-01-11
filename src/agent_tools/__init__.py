"""
Agent tools package.
"""

from .analysis import (
    load_audio,
    get_audio_info,
    extract_features,
    build_feature_vector,
    cosine_similarity,
    find_similar_tracks,
    compare_tracks,
    analyze_library,
    predict_genre_for_file,
    infer_key_from_features,
    infer_mood_from_features,
)
from .database import (
    get_db,
    init_database,
    db_add_track,
    db_store_embedding,
    db_update_embedding_metadata,
    db_get_track,
    db_get_all_tracks,
    db_get_all_embeddings,
    db_create_playlist,
    db_list_playlists,
    db_get_playlist,
    db_add_track_to_playlist,
    db_remove_track_from_playlist,
    db_delete_playlist,
    db_store_external_tracks,
    db_store_external_as_tracks,
)
from .visualization import create_visualizations, create_waveform_image, create_spectrogram_image
from .youtube import download_youtube_audio
from .reports import generate_report
from .api_clients import LLM, get_llm
from .converter import convert_audio
from .shazam import identify_track
from .deezer import get_music_trends
from .tagger import tag_audio
from .tts import text_to_speech

__all__ = [
    "load_audio",
    "get_audio_info",
    "extract_features",
    "build_feature_vector",
    "cosine_similarity",
    "find_similar_tracks",
    "compare_tracks",
    "analyze_library",
    "predict_genre_for_file",
    "infer_key_from_features",
    "infer_mood_from_features",
    "get_db",
    "init_database",
    "db_add_track",
    "db_store_embedding",
    "db_update_embedding_metadata",
    "db_get_track",
    "db_get_all_tracks",
    "db_get_all_embeddings",
    "db_create_playlist",
    "db_list_playlists",
    "db_get_playlist",
    "db_add_track_to_playlist",
    "db_remove_track_from_playlist",
    "db_delete_playlist",
    "db_store_external_tracks",
    "db_store_external_as_tracks",
    "create_visualizations",
    "create_waveform_image",
    "create_spectrogram_image",
    "download_youtube_audio",
    "generate_report",
    "LLM",
    "get_llm",
    "convert_audio",
    "identify_track",
    "get_music_trends",
    "tag_audio",
    "text_to_speech",
]
