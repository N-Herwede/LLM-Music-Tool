"""
Report generation.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import html as html_lib


def _rel_path(path: Union[str, Path], base: Path) -> str:
    try:
        return str(Path(path).relative_to(base)).replace("\\", "/")
    except ValueError:
        return Path(path).name


def generate_report(
    stats: Dict,
    plots: List[str],
    output_dir: Union[str, Path],
    highlights: Optional[Dict[str, Dict]] = None,
    track_sections: Optional[List[Dict]] = None,
    narrative: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Generate markdown + HTML reports, with optional PDF."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    file_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    total_tracks = stats.get("total_tracks", 0)
    genre_count = len(stats.get("genre_distribution", {}))

    genre_dist = stats.get("genre_distribution", {})
    tempo_stats = stats.get("tempo_stats", {})
    energy_stats = stats.get("energy_stats", {})
    top_genres = stats.get("top_genres", [])

    top_genre = top_genres[0] if top_genres else {}
    top_pct = top_genre.get("pct", 0)
    top_name = top_genre.get("genre", "unknown")
    top3_pct = sum(g.get("pct", 0) for g in top_genres[:3]) if top_genres else 0

    tempo_mean = tempo_stats.get("mean", 0)
    tempo_min = tempo_stats.get("min", 0)
    tempo_max = tempo_stats.get("max", 0)
    tempo_median = tempo_stats.get("median", 0)

    energy_mean = energy_stats.get("mean", 0)
    energy_median = energy_stats.get("median", 0)

    if tempo_mean >= 120:
        tempo_label = "fast-paced"
    elif tempo_mean <= 90:
        tempo_label = "slow-paced"
    else:
        tempo_label = "mid-tempo"

    if energy_mean >= 0.7:
        energy_label = "high energy"
    elif energy_mean <= 0.4:
        energy_label = "mostly mellow"
    else:
        energy_label = "balanced energy"

    report = f"""# Music Library Analysis Report

**Generated:** {timestamp}

## Overview

- **Total Tracks:** {total_tracks}
- **Genres:** {genre_count}
- **Average Tempo:** {stats.get('tempo_stats', {}).get('mean', 0):.1f} BPM
- **Tempo Spread (Std Dev):** {stats.get('tempo_stats', {}).get('std', 0):.1f} BPM
"""

    report += f"""
## Summary

Overall, the library is {tempo_label} with {energy_label}. The most common genre is
{top_name} ({top_pct:.1f}%), and the top 3 genres account for {top3_pct:.1f}% of all tracks.
Tempo ranges from {tempo_min:.1f} to {tempo_max:.1f} BPM, with a median of {tempo_median:.1f} BPM.
Energy has a median of {energy_median:.2f}.
"""
    if narrative:
        report += f"\n## LLM Narrative\n\n{narrative}\n"

    if highlights:
        report += "\n## Highlights\n\n"
        for label, item in highlights.items():
            title = item.get("title") or f"Track {item.get('id')}"
            genre = item.get("genre") or "unknown"
            tempo = item.get("tempo_bpm")
            energy = item.get("energy")
            bits = [title, f"genre: {genre}"]
            if tempo is not None:
                bits.append(f"tempo: {tempo:.1f} BPM")
            if energy is not None:
                bits.append(f"energy: {energy:.2f}")
            report += f"- **{label}:** " + ", ".join(bits) + "\n"

    report += """
## Genre Distribution

| Genre | Count | Percentage |
|-------|-------|------------|
"""

    for genre, count in sorted(genre_dist.items(), key=lambda x: -x[1]):
        pct = stats.get("genre_percentages", {}).get(genre, 0)
        report += f"| {genre.title()} | {count} | {pct:.1f}% |\n"

    report += f"""
## Tempo Analysis

| Statistic | Value |
|-----------|-------|
| Mean | {stats.get('tempo_stats', {}).get('mean', 0):.1f} BPM |
| Std Dev | {stats.get('tempo_stats', {}).get('std', 0):.1f} BPM |
| Min | {stats.get('tempo_stats', {}).get('min', 0):.1f} BPM |
| Max | {stats.get('tempo_stats', {}).get('max', 0):.1f} BPM |

## Average Tempo by Genre

| Genre | Tempo (BPM) |
|-------|-------------|
"""

    for genre, tempo in sorted(stats.get("avg_tempo_by_genre", {}).items()):
        report += f"| {genre.title()} | {tempo:.1f} |\n"

    report += "\n## Visualizations\n\n"
    for plot in plots:
        name = Path(plot).stem.replace("_", " ").title()
        report += f"### {name}\n![{name}]({_rel_path(plot, output_path)})\n\n"

    if track_sections:
        report += "## Track Visuals (Samples)\n\n"
        for section in track_sections:
            track = section.get("track", {})
            title = track.get("title") or f"Track {track.get('id')}"
            report += f"### {title}\n\n"
            report += f"- **Artist:** {track.get('artist') or 'Unknown'}\n"
            report += f"- **Genre:** {track.get('genre') or 'Unknown'}\n"
            if track.get("tempo_bpm") is not None:
                report += f"- **Tempo:** {track.get('tempo_bpm'):.1f} BPM\n"
            if track.get("energy") is not None:
                report += f"- **Energy:** {track.get('energy'):.2f}\n"
            for label, path in section.get("images", {}).items():
                report += f"\n**{label}:**\n![{label}]({_rel_path(path, output_path)})\n"
            report += "\n"

    md_path = output_path / f"report_{file_stamp}.md"
    md_path.write_text(report, encoding="utf-8")

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Music Report</title>
<style>
body{{font-family:Arial, sans-serif;color:#111;background:#f6f6f6;margin:0}}
main{{max-width:960px;margin:24px auto;background:#fff;padding:24px;border-radius:12px;box-shadow:0 6px 18px rgba(0,0,0,.08)}}
h1,h2,h3{{margin-top:1.2em}}
table{{border-collapse:collapse;width:100%;margin:12px 0}}
th,td{{border:1px solid #ddd;padding:8px;text-align:left}}
th{{background:#f0f0f0}}
img{{max-width:100%;border-radius:8px;border:1px solid #eee}}
.meta{{color:#666;font-size:.9em}}
.card{{border:1px solid #eee;border-radius:10px;padding:16px;margin:12px 0;background:#fafafa}}
</style>
</head>
<body><main>
<h1>Music Library Analysis Report</h1>
<p class="meta">Generated: {timestamp}</p>
<h2>Overview</h2>
<ul>
  <li>Total Tracks: {total_tracks}</li>
  <li>Genres: {genre_count}</li>
  <li>Average Tempo: {stats.get('tempo_stats', {}).get('mean', 0):.1f} BPM</li>
  <li>Tempo Spread (Std Dev): {stats.get('tempo_stats', {}).get('std', 0):.1f} BPM</li>
</ul>
<h2>Summary</h2>
<p>
Overall, the library is {tempo_label} with {energy_label}. The most common genre is
{top_name} ({top_pct:.1f}%), and the top 3 genres account for {top3_pct:.1f}% of all tracks.
Tempo ranges from {tempo_min:.1f} to {tempo_max:.1f} BPM, with a median of {tempo_median:.1f} BPM.
Energy has a median of {energy_median:.2f}.
</p>
"""
    if narrative:
        safe = html_lib.escape(narrative).replace("\n", "<br>")
        html_content += f"<h2>LLM Narrative</h2><p>{safe}</p>"

    if highlights:
        html_content += "<h2>Highlights</h2><ul>"
        for label, item in highlights.items():
            title = item.get("title") or f"Track {item.get('id')}"
            genre = item.get("genre") or "unknown"
            tempo = item.get("tempo_bpm")
            energy = item.get("energy")
            bits = [title, f"genre: {genre}"]
            if tempo is not None:
                bits.append(f"tempo: {tempo:.1f} BPM")
            if energy is not None:
                bits.append(f"energy: {energy:.2f}")
            html_content += f"<li><strong>{label}:</strong> " + ", ".join(bits) + "</li>"
        html_content += "</ul>"

    html_content += """
<h2>Genre Distribution</h2>
<table>
<thead><tr><th>Genre</th><th>Count</th><th>Percentage</th></tr></thead>
<tbody>
"""
    for genre, count in sorted(stats.get("genre_distribution", {}).items(), key=lambda x: -x[1]):
        pct = stats.get("genre_percentages", {}).get(genre, 0)
        html_content += f"<tr><td>{genre.title()}</td><td>{count}</td><td>{pct:.1f}%</td></tr>"

    html_content += "</tbody></table>"
    html_content += f"""
<h2>Tempo Analysis</h2>
<table>
<thead><tr><th>Statistic</th><th>Value</th></tr></thead>
<tbody>
<tr><td>Mean</td><td>{stats.get('tempo_stats', {}).get('mean', 0):.1f} BPM</td></tr>
<tr><td>Std Dev</td><td>{stats.get('tempo_stats', {}).get('std', 0):.1f} BPM</td></tr>
<tr><td>Min</td><td>{stats.get('tempo_stats', {}).get('min', 0):.1f} BPM</td></tr>
<tr><td>Max</td><td>{stats.get('tempo_stats', {}).get('max', 0):.1f} BPM</td></tr>
</tbody></table>
<h2>Average Tempo by Genre</h2>
<table>
<thead><tr><th>Genre</th><th>Tempo (BPM)</th></tr></thead>
<tbody>
"""
    for genre, tempo in sorted(stats.get("avg_tempo_by_genre", {}).items()):
        html_content += f"<tr><td>{genre.title()}</td><td>{tempo:.1f}</td></tr>"
    html_content += "</tbody></table>"

    html_content += "<h2>Visualizations</h2>"
    for plot in plots:
        name = Path(plot).stem.replace("_", " ").title()
        html_content += f"<div class='card'><h3>{name}</h3><img src='{_rel_path(plot, output_path)}' alt='{name}'></div>"

    if track_sections:
        html_content += "<h2>Track Visuals (Samples)</h2>"
        for section in track_sections:
            track = section.get("track", {})
            title = track.get("title") or f"Track {track.get('id')}"
            html_content += "<div class='card'>"
            html_content += f"<h3>{title}</h3>"
            html_content += "<ul>"
            html_content += f"<li>Artist: {track.get('artist') or 'Unknown'}</li>"
            html_content += f"<li>Genre: {track.get('genre') or 'Unknown'}</li>"
            if track.get("tempo_bpm") is not None:
                html_content += f"<li>Tempo: {track.get('tempo_bpm'):.1f} BPM</li>"
            if track.get("energy") is not None:
                html_content += f"<li>Energy: {track.get('energy'):.2f}</li>"
            html_content += "</ul>"
            for label, path in section.get("images", {}).items():
                html_content += f"<p><strong>{label}:</strong></p><img src='{_rel_path(path, output_path)}' alt='{label}'>"
            html_content += "</div>"

    html_content += "</main></body></html>"

    html_path = output_path / f"report_{file_stamp}.html"
    html_path.write_text(html_content, encoding="utf-8")

    pdf_path = None
    try:
        from weasyprint import HTML
        pdf_file = output_path / f"report_{file_stamp}.pdf"
        HTML(string=html_content, base_url=str(output_path)).write_pdf(str(pdf_file))
        pdf_path = str(pdf_file)
    except Exception:
        pdf_path = None

    return {
        "markdown_path": str(md_path),
        "html_path": str(html_path),
        "pdf_path": pdf_path,
    }


def generate_track_report(
    track: Dict[str, Any],
    features: Dict[str, Any],
    images: Dict[str, str],
    output_dir: Union[str, Path],
    summary: Optional[str] = None,
    narrative: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Generate a single-track markdown + HTML report, with optional PDF."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    file_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    title = track.get("title") or f"Track {track.get('id') or ''}".strip()
    artist = track.get("artist") or "Unknown"
    genre = track.get("genre") or "Unknown"
    tags = track.get("ai_tags") or []

    report = f"""# Track Report: {title}

**Generated:** {timestamp}

## Track Info

- **Title:** {title}
- **Artist:** {artist}
- **Genre:** {genre}
"""

    if track.get("tempo_bpm") is not None:
        report += f"- **Tempo:** {track.get('tempo_bpm'):.1f} BPM\n"
    if track.get("energy") is not None:
        report += f"- **Energy:** {track.get('energy'):.2f}\n"

    report += "\n## Audio Profile\n\n"
    report += (
        f"- **Spectral Centroid:** {features.get('spectral_centroid', 0):.0f}\n"
        f"- **Valence:** {features.get('valence', 0):.2f}\n"
    )
    if track.get("key"):
        report += f"- **Key:** {track.get('key')}\n"
    if track.get("mood"):
        report += f"- **Mood:** {track.get('mood')}\n"

    if summary:
        report += f"\n## Track Story\n\n{summary}\n"

    if narrative:
        report += f"\n## LLM Commentary\n\n{narrative}\n"

    if tags:
        report += "\n## AI Tags\n\n"
        for t in tags[:8]:
            report += f"- {t.get('tag')}: {t.get('score', 0):.2f}\n"

    report += "\n## Visuals\n\n"
    for label, path in images.items():
        report += f"### {label}\n![{label}]({_rel_path(path, output_path)})\n\n"

    md_path = output_path / f"track_report_{file_stamp}.md"
    md_path.write_text(report, encoding="utf-8")

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Track Report</title>
<style>
body{{font-family:Arial, sans-serif;color:#111;background:#f6f6f6;margin:0}}
main{{max-width:900px;margin:24px auto;background:#fff;padding:24px;border-radius:12px;box-shadow:0 6px 18px rgba(0,0,0,.08)}}
h1,h2,h3{{margin-top:1.2em}}
img{{max-width:100%;border-radius:8px;border:1px solid #eee}}
.meta{{color:#666;font-size:.9em}}
.card{{border:1px solid #eee;border-radius:10px;padding:16px;margin:12px 0;background:#fafafa}}
ul{{padding-left:18px}}
</style>
</head>
<body><main>
<h1>Track Report: {title}</h1>
<p class="meta">Generated: {timestamp}</p>
<h2>Track Info</h2>
<ul>
  <li>Title: {title}</li>
  <li>Artist: {artist}</li>
  <li>Genre: {genre}</li>
  <li>Tempo: {track.get('tempo_bpm', 0):.1f} BPM</li>
  <li>Energy: {track.get('energy', 0):.2f}</li>
</ul>
<h2>Audio Profile</h2>
<ul>
  <li>Spectral Centroid: {features.get('spectral_centroid', 0):.0f}</li>
  <li>Valence: {features.get('valence', 0):.2f}</li>
  <li>Key: {track.get('key') or 'Unknown'}</li>
  <li>Mood: {track.get('mood') or 'Unknown'}</li>
</ul>
"""
    if summary:
        html_content += f"<h2>Track Story</h2><p>{summary}</p>"

    if narrative:
        safe = html_lib.escape(narrative).replace("\n", "<br>")
        html_content += f"<h2>LLM Commentary</h2><p>{safe}</p>"

    if tags:
        html_content += "<h2>AI Tags</h2><ul>"
        for t in tags[:8]:
            html_content += f"<li>{t.get('tag')}: {t.get('score', 0):.2f}</li>"
        html_content += "</ul>"

    html_content += "<h2>Visuals</h2>"
    for label, path in images.items():
        html_content += f"<div class='card'><h3>{label}</h3><img src='{_rel_path(path, output_path)}' alt='{label}'></div>"

    html_content += "</main></body></html>"

    html_path = output_path / f"track_report_{file_stamp}.html"
    html_path.write_text(html_content, encoding="utf-8")

    pdf_path = None
    try:
        from weasyprint import HTML
        pdf_file = output_path / f"track_report_{file_stamp}.pdf"
        HTML(string=html_content, base_url=str(output_path)).write_pdf(str(pdf_file))
        pdf_path = str(pdf_file)
    except Exception:
        pdf_path = None

    return {
        "markdown_path": str(md_path),
        "html_path": str(html_path),
        "pdf_path": pdf_path,
    }
