"""
Visualization helpers.
"""

from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import matplotlib

matplotlib.use("Agg")


def create_visualizations(stats: Dict, output_dir: Union[str, Path]) -> List[str]:
    """Create visualization plots."""
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    import seaborn as sns

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plots = []

    plt.style.use("seaborn-v0_8-darkgrid")

    if stats.get("genre_distribution"):
        fig, ax = plt.subplots(figsize=(10, 8))
        genres = list(stats["genre_distribution"].keys())
        counts = list(stats["genre_distribution"].values())
        colors = [tuple(c) for c in colormaps["Set3"](np.linspace(0, 1, len(genres)))]

        ax.pie(counts, labels=genres, autopct="%1.1f%%", colors=colors, startangle=90)
        ax.set_title("Genre Distribution", fontsize=14, fontweight="bold")

        path = output_path / "genre_distribution.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        plots.append(str(path))

    if stats.get("avg_tempo_by_genre"):
        fig, ax = plt.subplots(figsize=(12, 6))
        genres = list(stats["avg_tempo_by_genre"].keys())
        tempos = list(stats["avg_tempo_by_genre"].values())

        bars = ax.bar(genres, tempos, color=colormaps["viridis"](np.linspace(0.2, 0.8, len(genres))))
        ax.set_xlabel("Genre", fontsize=12)
        ax.set_ylabel("Average Tempo (BPM)", fontsize=12)
        ax.set_title("Average Tempo by Genre", fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)

        for bar, tempo in zip(bars, tempos):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{tempo:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        path = output_path / "tempo_by_genre.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        plots.append(str(path))

    if stats.get("avg_energy_by_genre"):
        fig, ax = plt.subplots(figsize=(12, 6))
        genres = list(stats["avg_energy_by_genre"].keys())
        energy = list(stats["avg_energy_by_genre"].values())

        bars = ax.barh(genres, energy, color=colormaps["plasma"](np.linspace(0.2, 0.8, len(genres))))
        ax.set_xlabel("Average Energy", fontsize=12)
        ax.set_ylabel("Genre", fontsize=12)
        ax.set_title("Average Energy by Genre", fontsize=14, fontweight="bold")
        ax.set_xlim(0, 1.1)

        path = output_path / "energy_by_genre.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        plots.append(str(path))

    return plots


def create_waveform_image(filepath: Union[str, Path], output_dir: Union[str, Path]) -> str:
    """Create a waveform image and return its path."""
    import matplotlib.pyplot as plt
    import librosa

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    y, sr = librosa.load(str(filepath), sr=None, mono=True, duration=30)
    times = np.linspace(0, len(y) / sr, num=len(y), endpoint=False)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, y, linewidth=0.6, color="#4cc9f0")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform Preview", fontsize=12, fontweight="bold")
    ax.set_ylim(-1.0, 1.0)
    ax.grid(True, alpha=0.2)

    out_file = output_path / f"waveform_{Path(filepath).stem}.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(out_file)


def create_spectrogram_image(filepath: Union[str, Path], output_dir: Union[str, Path]) -> str:
    """Create a mel-spectrogram image and return its path."""
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    y, sr = librosa.load(str(filepath), sr=None, mono=True, duration=30)
    s = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    s_db = librosa.power_to_db(s, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 3))
    img = librosa.display.specshow(s_db, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
    ax.set_title("Mel Spectrogram", fontsize=12, fontweight="bold")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    out_file = output_path / f"spectrogram_{Path(filepath).stem}.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(out_file)
