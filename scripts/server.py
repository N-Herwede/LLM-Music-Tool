#!/usr/bin/env python3
"""
Music Agent Pro - Web Server
============================

Flask web interface for the Music Agent.

Usage:
    python scripts/server.py
    
Then open: http://localhost:5000
"""

import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flask import Flask, render_template_string, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from agent import Agent
from agent_tools.api_clients import get_llm, Ollama, Groq, OpenAI, Anthropic

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# Directories
UPLOAD_DIR = ROOT / "data" / "uploads"
CONVERT_DIR = ROOT / "data" / "processed" / "converted"
REPORTS_DIR = ROOT / "data" / "processed" / "reports"
TTS_DIR = ROOT / "data" / "processed" / "tts"
YOUTUBE_DIR = ROOT / "data" / "raw" / "youtube"

# Ensure directories exist
for d in [UPLOAD_DIR, CONVERT_DIR, REPORTS_DIR, TTS_DIR, YOUTUBE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Global agent instance
agent = None
current_llm_name = None


def get_available_llms():
    """Get list of available LLM providers."""
    available = []
    for name, cls in [("ollama", Ollama), ("groq", Groq), ("openai", OpenAI), ("anthropic", Anthropic)]:
        try:
            instance = cls()
            if instance.available():
                available.append({"name": name, "display": instance.name})
        except:
            pass
    if not available:
        available.append({"name": "none", "display": "Fallback (no LLM)"})
    return available


def get_agent(llm_name=None):
    """Get or create the agent instance."""
    global agent, current_llm_name
    
    if llm_name and llm_name != current_llm_name:
        # Switch LLM
        agent = None
        current_llm_name = llm_name
    
    if agent is None:
        llm = get_llm(current_llm_name)
        current_llm_name = llm_name or (llm.name.split("/")[0].lower() if llm else "none")
        print(f"LLM: {llm.name if llm else 'Fallback mode (no LLM)'}")
        agent = Agent(llm)
    
    return agent


# =============================================================================
# HTML TEMPLATE
# =============================================================================

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Music Agent</title>
<style>
:root {
    --bg: #1a2a3a;
    --bg2: #203548;
    --panel: #f6f7fb;
    --panel-shadow: rgba(0, 0, 0, 0.18);
    --text: #e9f0f7;
    --text-dark: #1f2a33;
    --muted: #a8b6c4;
    --accent: #4ea7ff;
    --accent-strong: #2b7fe6;
    --border: rgba(0, 0, 0, 0.08);
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: "Space Grotesk", "Sora", "IBM Plex Sans", "Segoe UI", sans-serif;
    background:
        radial-gradient(1200px 600px at -10% 0%, rgba(78, 167, 255, 0.25), transparent 60%),
        radial-gradient(900px 500px at 110% 20%, rgba(40, 82, 130, 0.6), transparent 55%),
        linear-gradient(160deg, #1a2a3a 0%, #1d3448 45%, #152432 100%);
    color: var(--text);
    min-height: 100vh;
}

.shell {
    max-width: 1200px;
    margin: 0 auto;
    padding: 28px 20px 36px;
    min-height: 100vh;
    height: 100vh;
    display: flex;
    flex-direction: column;
    gap: 18px;
}

.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
}

.brand {
    font-size: 1.35rem;
    letter-spacing: 0.02em;
    font-weight: 500;
}

.brand span {
    opacity: 0.85;
    font-weight: 400;
    font-size: 0.95rem;
    margin-left: 12px;
}

.llm-selector {
    display: flex;
    align-items: center;
    gap: 10px;
    background: rgba(8, 15, 24, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 8px 12px;
    border-radius: 12px;
}

.llm-selector label {
    color: var(--muted);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
}

.llm-selector select {
    background: rgba(12, 20, 30, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: var(--text);
    border-radius: 10px;
    padding: 6px 10px;
    font-size: 0.85rem;
}

.llm-status {
    font-size: 0.75rem;
    color: var(--muted);
}

.workspace {
    display: grid;
    grid-template-columns: minmax(340px, 1fr) minmax(420px, 1.3fr);
    grid-template-areas: "sidebar response";
    gap: 20px;
    flex: 1;
    min-height: 0;
}

.panel {
    background: var(--panel);
    color: var(--text-dark);
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 18px 50px var(--panel-shadow);
    display: flex;
    flex-direction: column;
    gap: 16px;
    min-height: 0;
}

.panel-title {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: rgba(31, 42, 51, 0.55);
}

.quick-actions {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.qa-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 8px;
}

.quick-actions button {
    border: 1px solid rgba(0, 0, 0, 0.08);
    background: #ffffff;
    padding: 10px 12px;
    border-radius: 10px;
    font-size: 0.82rem;
    text-align: left;
    cursor: pointer;
    transition: all 0.2s ease;
}

.quick-actions button:hover {
    border-color: rgba(78, 167, 255, 0.6);
    box-shadow: 0 6px 16px rgba(78, 167, 255, 0.2);
    transform: translateY(-1px);
}

.messages {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding-right: 4px;
}

.msg {
    max-width: 90%;
    padding: 12px 16px;
    border-radius: 12px;
    line-height: 1.6;
    white-space: pre-wrap;
    font-size: 0.92rem;
}

.msg.user {
    align-self: flex-end;
    background: rgba(78, 167, 255, 0.15);
    border: 1px solid rgba(78, 167, 255, 0.35);
}

.msg.bot {
    align-self: flex-start;
    background: #ffffff;
    border: 1px solid rgba(0, 0, 0, 0.06);
}

.msg strong { color: #2b7fe6; }
.msg a { color: #2b7fe6; }

.input-area {
    display: flex;
    gap: 10px;
    align-items: center;
    border-top: 1px solid rgba(0, 0, 0, 0.08);
    padding-top: 12px;
}

.input-area input[type="text"] {
    flex: 1;
    border: 1px solid rgba(0, 0, 0, 0.12);
    border-radius: 10px;
    padding: 10px 12px;
    font-size: 0.9rem;
}

.btn {
    background: var(--accent);
    border: none;
    border-radius: 10px;
    padding: 10px 16px;
    font-weight: 600;
    color: #fff;
    cursor: pointer;
}

.btn:hover { background: var(--accent-strong); }
.btn:disabled { background: rgba(0, 0, 0, 0.25); cursor: not-allowed; }

.file-upload {
    position: relative;
}

.file-upload input[type="file"] {
    position: absolute;
    opacity: 0;
    width: 40px;
    height: 40px;
    cursor: pointer;
}

.file-icon {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    border: 1px dashed rgba(0, 0, 0, 0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    color: rgba(31, 42, 51, 0.7);
    background: rgba(78, 167, 255, 0.08);
}

.filename {
    font-size: 0.75rem;
    color: rgba(31, 42, 51, 0.55);
    max-width: 140px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.report-frame,
.report-image {
    width: 100%;
    border: 1px solid rgba(0, 0, 0, 0.08);
    border-radius: 12px;
    margin-top: 12px;
    background: #fff;
}

.report-frame {
    height: 360px;
}

@media (max-width: 980px) {
    .workspace {
        grid-template-columns: 1fr;
        grid-template-areas:
            "sidebar"
            "response";
    }
}

.section-group {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.section-title {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: rgba(31, 42, 51, 0.55);
    margin-top: 8px;
}

.input-wrap {
    background: #ffffff;
    border: 1px solid rgba(0, 0, 0, 0.08);
    border-radius: 12px;
    padding: 12px;
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.08);
}

.input-area {
    border-top: none;
    padding-top: 0;
}

.sidebar {
    display: flex;
    flex-direction: column;
    gap: 16px;
    grid-area: sidebar;
}

.response-panel {
    grid-area: response;
}

.panel.input-panel {
    padding: 14px 16px 16px;
}

.panel.input-panel .panel-title {
    margin-bottom: 4px;
}

.tool-catalog details {
    border: 1px solid rgba(0, 0, 0, 0.08);
    border-radius: 10px;
    background: #ffffff;
    padding: 10px 12px;
}

.tool-catalog summary {
    cursor: pointer;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: rgba(31, 42, 51, 0.55);
    list-style: none;
}

.tool-catalog summary::-webkit-details-marker {
    display: none;
}

.tool-list {
    margin-top: 10px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 8px 12px;
    padding-left: 0;
    list-style: none;
}

.tool-list li {
    font-size: 0.82rem;
    color: rgba(31, 42, 51, 0.75);
    background: rgba(78, 167, 255, 0.08);
    border-radius: 8px;
    padding: 8px 10px;
}

.tool-list strong {
    color: rgba(31, 42, 51, 0.9);
    font-weight: 600;
}
</style>
</head>
<body>
<div class="shell">
    <div class="topbar">
        <div class="brand">Music Agent <span>Workspace</span></div>
        <div class="llm-selector">
            <label>LLM</label>
            <select id="llmSelect">
                <option value="">Loading...</option>
            </select>
            <span class="llm-status" id="llmStatus"></span>
        </div>
    </div>

    <div class="workspace">
        <aside class="sidebar">
            <div class="panel">
            <div class="panel-title">Prompt</div>
            <div class="quick-actions">
                <div class="section-group">
                    <div class="section-title">Demo flow (crescendo)</div>
                    <div class="qa-grid">
                        <button data-prompt="Analyze my library">1. Library overview</button>
                        <button data-prompt="With the uploaded file, generate a waveform preview">2. Waveform preview (uploaded)</button>
                        <button data-prompt="With the uploaded file, generate a track report">3. Track report (uploaded)</button>
                        <button data-prompt="Generate a full library report">4. Full library report</button>
                        <button data-prompt="With the uploaded file, convert to wav">5. Convert to WAV (uploaded)</button>
                        <button data-prompt="Show Deezer trends in FR">6. Deezer trends (country)</button>
                        <button data-prompt="Find similar to track 1">7. Similarity search</button>
                    </div>
                </div>
                <div class="section-group">
                    <div class="section-title">Extra capabilities</div>
                    <div class="qa-grid">
                        <button data-prompt="With the uploaded file, detect genre">Genre detection (uploaded)</button>
                        <button data-prompt="With the uploaded file, tag audio">Tag audio (uploaded)</button>
                        <button data-prompt="With the uploaded file, identify this song">Shazam identify (uploaded)</button>
                        <button data-prompt="Playlist for workout">Playlist generation</button>
                        <button data-prompt="Tempo between 120 and 140">Tempo filter</button>
                        <button data-prompt="Convert this text to speech: The library is ready for analysis.">Text to speech</button>
                    </div>
                </div>
            </div>
        </div>
            <div class="tool-catalog">
                <details>
                    <summary>Tool catalog</summary>
                    <ul class="tool-list">
                        <li><strong>analyze_library</strong> - library stats overview</li>
                        <li><strong>generate_report</strong> - report for track or library</li>
                        <li><strong>recommend_tracks</strong> - similar tracks search</li>
                        <li><strong>compare_tracks</strong> - compare two tracks</li>
                        <li><strong>find_by_mood</strong> - mood-based selection</li>
                        <li><strong>filter_by_tempo</strong> - BPM range filter</li>
                        <li><strong>generate_playlist</strong> - themed playlist</li>
                        <li><strong>get_track_info</strong> - track metadata</li>
                        <li><strong>generate_waveform</strong> - waveform preview</li>
                        <li><strong>detect_genre</strong> - genre prediction</li>
                        <li><strong>tag_audio</strong> - audio tagging</li>
                        <li><strong>identify_track</strong> - Shazam identification</li>
                        <li><strong>analyze_upload</strong> - analyze uploaded file</li>
                        <li><strong>save_upload</strong> - save upload to DB</li>
                        <li><strong>convert_audio</strong> - audio conversion</li>
                        <li><strong>download_youtube_audio</strong> - YouTube ingest</li>
                        <li><strong>get_music_trends</strong> - Deezer trends</li>
                        <li><strong>text_to_speech</strong> - TTS audio</li>
                        <li><strong>create_playlist</strong> - create playlist</li>
                        <li><strong>list_playlists</strong> - list playlists</li>
                        <li><strong>get_playlist</strong> - playlist details</li>
                        <li><strong>add_to_playlist</strong> - append track</li>
                        <li><strong>remove_from_playlist</strong> - remove by position</li>
                        <li><strong>delete_playlist</strong> - delete playlist</li>
                    </ul>
                </details>
            </div>
        <div class="panel input-panel">
            <div class="panel-title">Message</div>
            <div class="input-wrap">
                <div class="input-area">
                    <div class="file-upload">
                        <input type="file" id="fileInput" accept=".wav,.mp3,.au,.flac,.ogg">
                        <div class="file-icon">&#127925;</div>
                    </div>
                    <span class="filename" id="filename"></span>
                    <input type="text" id="textInput" placeholder="Type a message...">
                    <button class="btn" id="sendBtn">Send</button>
                </div>
            </div>
        </div>
    </aside>

        <section class="panel response-panel">
            <div class="panel-title">Response</div>
            <div class="messages" id="messages">
                <div class="msg bot">Welcome! Ask me anything about your music library, or <strong>upload an audio file</strong> to analyze it!</div>
            </div>
        </section>
    </div>
</div>
<script>
window.addEventListener('DOMContentLoaded', () => {
    const messagesDiv = document.getElementById('messages');
    const textInput = document.getElementById('textInput');
    const sendBtn = document.getElementById('sendBtn');
    const fileInput = document.getElementById('fileInput');
    const filenameSpan = document.getElementById('filename');
    const promptButtons = document.querySelectorAll('[data-prompt]');
    const llmSelect = document.getElementById('llmSelect');
    const llmStatus = document.getElementById('llmStatus');

    let selectedFile = null;

    function addMessage(text, isUser) {
        const div = document.createElement('div');
        div.className = 'msg ' + (isUser ? 'user' : 'bot');
        div.innerHTML = formatMessage(text);
        messagesDiv.appendChild(div);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    function formatMessage(text) {
        const reportsKey = '/data/processed/reports/';
        const convertedKey = '/data/processed/converted/';
        const ttsKey = '/data/processed/tts/';
        const youtubeKey = '/data/raw/youtube/';

        const normalizeToken = (token) => {
            let clean = token;
            while (clean.length && '.,)'.includes(clean[clean.length - 1])) {
                clean = clean.slice(0, -1);
            }
            return clean;
        };

        const mapToWebPath = (token) => {
            const normalized = token.replaceAll('\\\\', '/');
            const lower = normalized.toLowerCase();
            if (lower.includes(reportsKey)) {
                return '/reports/' + normalized.split(reportsKey)[1];
            }
            if (lower.includes(convertedKey)) {
                return '/download/' + normalized.split(convertedKey)[1];
            }
            if (lower.includes(ttsKey)) {
                return '/tts/' + normalized.split(ttsKey)[1];
            }
            if (lower.includes(youtubeKey)) {
                return '/downloads/youtube/' + normalized.split(youtubeKey)[1];
            }
            return null;
        };

        const linkifyToken = (token) => {
            const clean = normalizeToken(token);
            const webPath = mapToWebPath(clean);
            if (webPath) {
                const safePath = encodeURI(webPath);
                return '<a href="' + safePath + '" target="_blank">' + safePath + '</a>';
            }
            if (clean.startsWith('http://') || clean.startsWith('https://')) {
                return '<a href="' + clean + '" target="_blank">' + clean + '</a>';
            }
            if (clean.startsWith('/reports/') || clean.startsWith('/download/') || clean.startsWith('/downloads/') || clean.startsWith('/tts/')) {
                return '<a href="' + clean + '" target="_blank">' + clean + '</a>';
            }
            return token;
        };

        const rawText = text;
        const normalizedText = text.replaceAll('\\n', ' ').replaceAll('\\r', ' ').replaceAll('\\t', ' ');
        const tokens = normalizedText.split(' ').filter((token) => token.length > 0);
        let formatted = tokens.map(linkifyToken).join(' ');

        const findReportPath = (exts) => {
            let idx = rawText.indexOf('/reports/');
            while (idx !== -1) {
                let end = idx;
                while (end < rawText.length) {
                    const ch = rawText[end];
                    if (ch === ' ' || ch === '\\n' || ch === '\\r' || ch === '\\t') {
                        break;
                    }
                    end += 1;
                }
                let token = rawText.slice(idx, end).replaceAll('\"', '');
                token = normalizeToken(token);
                const lower = token.toLowerCase();
                if (exts.some((ext) => lower.endsWith(ext))) {
                    return token;
                }
                idx = rawText.indexOf('/reports/', idx + 1);
            }
            return null;
        };

        const htmlPreview = findReportPath(['.html']);
        if (htmlPreview) {
            formatted += '<iframe class="report-frame" src="' + htmlPreview + '"></iframe>';
        }
        const imagePreview = findReportPath(['.png', '.jpg', '.jpeg', '.gif']);
        if (imagePreview) {
            formatted += '<img class="report-image" src="' + imagePreview + '" alt="Report image">';
        }

        return formatted;
    }

    function fileSelected(event) {
        const input = event.target;
        if (input.files.length > 0) {
            selectedFile = input.files[0];
            filenameSpan.textContent = selectedFile.name;
        }
    }

    async function sendMessage() {
        const text = textInput.value.trim();
        if (!text && !selectedFile) return;

        // Show user message
        const displayText = selectedFile ? `File: ${selectedFile.name}${text ? ': ' + text : ''}` : text;
        addMessage(displayText, true);

        textInput.value = '';
        sendBtn.disabled = true;

        try {
            let response;

            if (selectedFile) {
                const formData = new FormData();
                formData.append('file', selectedFile);
                formData.append('msg', text);
                response = await fetch('/upload', { method: 'POST', body: formData });
                selectedFile = null;
                filenameSpan.textContent = '';
                fileInput.value = '';
            } else {
                response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ msg: text })
                });
            }

            const data = await response.json();
            addMessage(data.response || data.r || 'No response', false);
        } catch (error) {
            addMessage('Error: ' + error.message, false);
        }

        sendBtn.disabled = false;
        textInput.focus();
    }

    async function loadLLMs() {
        try {
            const response = await fetch('/api/llms');
            const data = await response.json();

            if (!llmSelect || !llmStatus) {
                return;
            }

            llmSelect.innerHTML = '';
            data.llms.forEach((llm) => {
                const option = document.createElement('option');
                option.value = llm.name;
                option.textContent = llm.display;
                if (llm.name === data.current) {
                    option.selected = true;
                }
                llmSelect.appendChild(option);
            });

            llmStatus.textContent = 'Ready';
            llmStatus.style.color = '#238636';
        } catch (error) {
            if (llmStatus) {
                llmStatus.textContent = 'Error';
                llmStatus.style.color = '#f85149';
            }
        }
    }

    async function switchLLM(llmName) {
        if (!llmStatus) return;
        llmStatus.textContent = 'Switching...';
        llmStatus.style.color = '#d29922';

        try {
            const response = await fetch('/api/llm/switch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ llm: llmName })
            });
            const data = await response.json();

            if (data.ok) {
                llmStatus.textContent = data.display;
                llmStatus.style.color = '#238636';
                addMessage('Switched to ' + data.display, false);
            } else {
                llmStatus.textContent = data.error;
                llmStatus.style.color = '#f85149';
            }
        } catch (error) {
            llmStatus.textContent = 'Error';
            llmStatus.style.color = '#f85149';
        }
    }

    if (fileInput) {
        fileInput.addEventListener('change', fileSelected);
    }

    if (sendBtn && textInput) {
        sendBtn.addEventListener('click', sendMessage);
        textInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                event.preventDefault();
                sendMessage();
            }
        });
    }

    promptButtons.forEach((button) => {
        button.addEventListener('click', () => {
            if (!textInput) return;
            const prompt = button.getAttribute('data-prompt') || '';
            textInput.value = prompt;
            textInput.focus();
        });
    });

    if (llmSelect) {
        llmSelect.addEventListener('change', (event) => {
            switchLLM(event.target.value);
        });
    }

    loadLLMs();
    if (textInput) {
        textInput.focus();
    }
});
</script>
</body>
</html>'''


# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    """Serve the main page."""
    a = get_agent()
    llm_name = a.llm.name if a.llm else "Fallback Mode"
    return render_template_string(HTML_TEMPLATE, llm=llm_name)


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    data = request.get_json(silent=True) or {}
    msg = data.get('msg', '')
    
    response = get_agent().chat(msg)
    return jsonify({'response': response})


@app.route('/upload', methods=['POST'])
def upload():
    """Handle file uploads."""
    if 'file' not in request.files:
        return jsonify({'response': 'Error: No file provided'})
    
    file = request.files['file']
    if not file.filename:
        return jsonify({'response': 'Error: Empty filename'})
    
    # Save file
    filename = secure_filename(file.filename)
    filepath = UPLOAD_DIR / filename
    file.save(filepath)
    
    # Process with agent
    msg = request.form.get('msg', '')
    response = get_agent().chat(msg, str(filepath))
    
    # Clean up uploaded file
    try:
        filepath.unlink()
    except Exception:
        pass
    
    return jsonify({'response': response})


@app.route('/download/<path:filename>')
def download(filename):
    """Serve converted audio files."""
    return send_from_directory(CONVERT_DIR, filename, as_attachment=True)


@app.route('/reports/<path:filename>')
def reports(filename):
    """Serve report files."""
    return send_from_directory(REPORTS_DIR, filename)


@app.route('/downloads/youtube/<path:filename>')
def downloads_youtube(filename):
    """Serve YouTube downloads."""
    return send_from_directory(YOUTUBE_DIR, filename, as_attachment=True)


@app.route('/tts/<path:filename>')
def tts(filename):
    """Serve TTS audio files."""
    return send_from_directory(TTS_DIR, filename, as_attachment=True)


@app.route('/api/llms')
def list_llms():
    """List available LLM providers."""
    available = get_available_llms()
    return jsonify({
        'llms': available,
        'current': current_llm_name or (available[0]['name'] if available else 'none')
    })


@app.route('/api/llm/switch', methods=['POST'])
def switch_llm():
    """Switch LLM provider."""
    global agent, current_llm_name
    data = request.get_json(silent=True) or {}
    llm_name = data.get('llm')
    
    if not llm_name:
        return jsonify({'ok': False, 'error': 'Missing llm parameter'})
    
    try:
        agent = None  # Reset agent
        get_agent(llm_name)
        return jsonify({
            'ok': True, 
            'llm': current_llm_name,
            'display': agent.llm.name if agent and agent.llm else 'Fallback'
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Start the server."""
    print("Music Agent Pro")
    
    # Check database
    db_path = ROOT / "data" / "processed" / "music_library.db"
    if not db_path.exists():
        print("\nDatabase not found!")
        print("Run: python scripts/setup_database.py")
        print()
    else:
        print(f"Database: {db_path}")
    
    # Initialize agent
    get_agent()
    
    print("\nStarting server...")
    print("   http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    main()
