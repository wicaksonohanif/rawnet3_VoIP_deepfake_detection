# streamlit_app.py — VoIP/WhatsApp Deepfake Detector (Enhanced UI)
# Jalankan: streamlit run streamlit_app.py
#
# Requirements:
# pip install streamlit torch numpy librosa matplotlib soundfile scipy audioread

import streamlit as st
import torch
import numpy as np

# Try importing audio libraries with helpful error messages
try:
    import librosa
    import librosa.display
except ImportError:
    st.error("⚠️ `librosa` not installed. Run: `pip install librosa`")
    st.stop()

try:
    import soundfile
except ImportError:
    st.warning("⚠️ `soundfile` not installed. Some audio formats may not work. Run: `pip install soundfile`")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tempfile, os, time
from pathlib import Path

try:
    from inference import preprocess_for_inference, predict, load_model
except ImportError:
    st.error("⚠️ Cannot import `inference` module. Ensure `inference.py` is in the same directory.")
    st.stop()

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VoiceGuard — Deepfake Detector",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

/* ─── Reset & Base ─── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ─── App Background ─── */
.stApp {
    background: #06070D;
    color: #E0E6F0;
}

.main .block-container {
    padding: 2.5rem 2rem 4rem;
    max-width: 800px;
}

/* ─── Hero Header ─── */
.hero-container {
    text-align: center;
    padding: 2.5rem 0 2rem;
    position: relative;
}

.hero-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: #00E5CC;
    background: rgba(0, 229, 204, 0.08);
    border: 1px solid rgba(0, 229, 204, 0.25);
    border-radius: 20px;
    padding: 0.3rem 1rem;
    margin-bottom: 1.2rem;
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin: 0 0 0.6rem;
    background: linear-gradient(135deg, #FFFFFF 0%, #A8B8D0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-size: 1rem;
    color: #6B7A8F;
    font-weight: 400;
    margin-bottom: 0;
    letter-spacing: 0.01em;
}

/* ─── Scan Line Animation ─── */
@keyframes scan {
    0%   { transform: translateY(-100%); opacity: 0; }
    10%  { opacity: 0.4; }
    90%  { opacity: 0.4; }
    100% { transform: translateY(400%); opacity: 0; }
}

.scanline {
    position: absolute;
    left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00E5CC, transparent);
    animation: scan 3s ease-in-out infinite;
    pointer-events: none;
}

/* ─── Pipeline Info Card ─── */
.pipeline-card {
    background: #0E1220;
    border: 1px solid #1E2840;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin: 1.5rem 0;
    display: flex;
    gap: 1rem;
    align-items: flex-start;
}

.pipeline-icon {
    font-size: 1.3rem;
    margin-top: 0.1rem;
}

.pipeline-title {
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    color: #00E5CC;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.5rem;
}

.pipeline-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 0.75rem;
}

.pipeline-item {
    background: rgba(255,255,255,0.03);
    border: 1px solid #1E2840;
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
}

.pipeline-label {
    font-size: 0.7rem;
    color: #4A5568;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.2rem;
}

.pipeline-value {
    font-size: 0.85rem;
    font-weight: 600;
    color: #CBD5E0;
    font-family: 'JetBrains Mono', monospace;
}

/* ─── Status Bar ─── */
.status-bar {
    background: #0E1220;
    border: 1px solid #1E2840;
    border-radius: 10px;
    padding: 0.75rem 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #4A5568;
    margin-bottom: 1.5rem;
}

.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #00E5CC;
    box-shadow: 0 0 8px #00E5CC;
    flex-shrink: 0;
}

.status-dot.warn { background: #F6C90E; box-shadow: 0 0 8px #F6C90E; }
.status-dot.error { background: #FF4D6D; box-shadow: 0 0 8px #FF4D6D; }

.status-text { color: #8899AA; }
.status-text span { color: #00E5CC; }

/* ─── Section Headers ─── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 2rem 0 1rem;
}

.section-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1E2840, transparent);
}

.section-label {
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    color: #4A5568;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* ─── Network Selector ─── */
.stSelectbox > div > div {
    background: #0E1220 !important;
    border: 1px solid #1E2840 !important;
    border-radius: 10px !important;
    color: #CBD5E0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

.stSelectbox > div > div:hover {
    border-color: #00E5CC !important;
}

/* ─── Upload Zone ─── */
[data-testid="stFileUploader"] {
    background: #0E1220;
    border: 2px dashed #1E2840;
    border-radius: 14px;
    padding: 1rem;
    transition: border-color 0.3s;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(0, 229, 204, 0.4);
}

[data-testid="stFileUploader"] label {
    color: #6B7A8F !important;
}

/* ─── Waveform Container ─── */
.waveform-container {
    background: #0A0D16;
    border: 1px solid #1E2840;
    border-radius: 14px;
    padding: 1rem;
    margin: 1rem 0;
}

/* ─── Analyze Button ─── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #00B4D8, #00E5CC) !important;
    color: #06070D !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    text-transform: uppercase !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(0, 229, 204, 0.3) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* ─── Result Cards ─── */
.result-card {
    border-radius: 16px;
    padding: 1.75rem 2rem;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
}

.result-card.bonafide {
    background: linear-gradient(135deg, #071A12 0%, #0A2018 100%);
    border: 1px solid rgba(0, 200, 100, 0.3);
}

.result-card.spoofed {
    background: linear-gradient(135deg, #1A0710 0%, #200A0E 100%);
    border: 1px solid rgba(255, 60, 80, 0.3);
}

.result-glow-b {
    position: absolute;
    top: -40px; right: -40px;
    width: 120px; height: 120px;
    background: rgba(0, 200, 100, 0.12);
    border-radius: 50%;
    filter: blur(30px);
}

.result-glow-s {
    position: absolute;
    top: -40px; right: -40px;
    width: 120px; height: 120px;
    background: rgba(255, 60, 80, 0.12);
    border-radius: 50%;
    filter: blur(30px);
}

.result-verdict {
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    margin-bottom: 0.3rem;
}

.result-verdict.bonafide { color: #00E87A; }
.result-verdict.spoofed  { color: #FF4D6D; }

.result-desc {
    font-size: 0.9rem;
    color: #8899AA;
}

.result-confidence {
    position: absolute;
    top: 1.75rem; right: 2rem;
    text-align: right;
}

.confidence-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
}
.confidence-num.bonafide { color: #00E87A; }
.confidence-num.spoofed  { color: #FF4D6D; }

.confidence-label {
    font-size: 0.7rem;
    color: #4A5568;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}

/* ─── Probability Bars ─── */
.prob-container {
    background: #0E1220;
    border: 1px solid #1E2840;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-top: 1rem;
}

.prob-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1rem;
}

.prob-row:last-child { margin-bottom: 0; }

.prob-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #8899AA;
    width: 90px;
    flex-shrink: 0;
}

.prob-bar-track {
    flex: 1;
    height: 8px;
    background: #0A0D16;
    border-radius: 4px;
    overflow: hidden;
}

.prob-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

.prob-bar-fill.bonafide { background: linear-gradient(90deg, #00A86B, #00E87A); }
.prob-bar-fill.spoofed  { background: linear-gradient(90deg, #C0203A, #FF4D6D); }

.prob-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 700;
    width: 52px;
    text-align: right;
}
.prob-value.bonafide { color: #00E87A; }
.prob-value.spoofed  { color: #FF4D6D; }

/* ─── Metric Chips ─── */
.metrics-row {
    display: flex;
    gap: 0.75rem;
    margin-top: 1.25rem;
}

.metric-chip {
    flex: 1;
    background: #0A0D16;
    border: 1px solid #1E2840;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    text-align: center;
}

.metric-chip-label {
    font-size: 0.68rem;
    color: #4A5568;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.3rem;
}

.metric-chip-value {
    font-size: 1.1rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    color: #CBD5E0;
}

/* ─── Matplotlib styles ─── */
[data-testid="stImage"] { border-radius: 12px; overflow: hidden; }

/* ─── Audio player ─── */
[data-testid="stAudio"] {
    background: #0E1220 !important;
    border-radius: 10px !important;
    padding: 0.5rem !important;
}

/* ─── Spinner ─── */
.stSpinner { color: #00E5CC !important; }

/* ─── Footer ─── */
.footer {
    text-align: center;
    padding: 2rem 0 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #2A3448;
    letter-spacing: 0.05em;
}

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #06070D; }
::-webkit-scrollbar-thumb { background: #1E2840; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2A3A58; }
</style>
""", unsafe_allow_html=True)


# ── Helper: HTML Section Divider ─────────────────────────────────────────────
def section_header(label):
    st.markdown(f"""
    <div class="section-header">
        <div class="section-line"></div>
        <span class="section-label">{label}</span>
        <div class="section-line" style="background: linear-gradient(90deg, transparent, #1E2840);"></div>
    </div>
    """, unsafe_allow_html=True)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="scanline"></div>
    <div class="hero-badge">⬡ VOICE FORENSICS ENGINE v2.1</div>
    <div class="hero-title">VoiceGuard</div>
    <div class="hero-sub">Neural Deepfake Detection · RawNet3 Architecture · VoIP/WhatsApp Pipeline</div>
</div>
""", unsafe_allow_html=True)


# ── Pipeline Info Card ────────────────────────────────────────────────────────
st.markdown("""
<div class="pipeline-card">
    <div class="pipeline-icon">⚙️</div>
    <div style="flex:1">
        <div class="pipeline-title">DETECTION PIPELINE</div>
        <div class="pipeline-grid">
            <div class="pipeline-item">
                <div class="pipeline-label">Bandwidth</div>
                <div class="pipeline-value">50–7000 Hz</div>
            </div>
            <div class="pipeline-item">
                <div class="pipeline-label">Codec</div>
                <div class="pipeline-value">Opus WB</div>
            </div>
            <div class="pipeline-item">
                <div class="pipeline-label">Noise Floor</div>
                <div class="pipeline-value">SNR 25 dB</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, cfg = load_model('rawnet3_voip_weights.pt', 'config.json', device)
    return model, cfg, device

with st.spinner("Initializing neural network..."):
    model, cfg, device = get_model()

device_icon  = "⚡" if device == 'cuda' else "🖥"
device_label = "GPU — CUDA" if device == 'cuda' else "CPU — Inference"

st.markdown(f"""
<div class="status-bar">
    <div class="status-dot"></div>
    <span class="status-text">
        Model ready &nbsp;·&nbsp;
        <span>{device_icon} {device_label}</span> &nbsp;·&nbsp;
        Dataset: <span>ASVspoof 2021 DF</span> &nbsp;·&nbsp;
        Architecture: <span>RawNet3-GatedConvNext</span>
    </span>
</div>
""", unsafe_allow_html=True)


# ── Network Condition ─────────────────────────────────────────────────────────
section_header("01 · NETWORK CONDITION")

col_net, col_info = st.columns([2, 1])
with col_net:
    bitrate = st.selectbox(
        "Opus Bitrate Simulation",
        ['high (wifi/4G baik)', 'medium (normal)', 'low (2G/buruk)'],
        index=1,
        label_visibility="collapsed"
    )
bm = bitrate.split()[0]

bitrate_details = {
    'high':   ("≥ 32 kbps", "#00E5CC"),
    'medium': ("16–24 kbps", "#F6C90E"),
    'low':    ("6–12 kbps",  "#FF4D6D"),
}
bw_val, bw_col = bitrate_details[bm]
with col_info:
    st.markdown(f"""
    <div style="background:#0E1220; border:1px solid #1E2840; border-radius:10px;
                padding:0.6rem 1rem; font-family:'JetBrains Mono',monospace;">
        <div style="font-size:0.68rem; color:#4A5568; text-transform:uppercase; letter-spacing:0.08em;">Bitrate</div>
        <div style="font-size:1rem; font-weight:700; color:{bw_col};">{bw_val}</div>
    </div>
    """, unsafe_allow_html=True)


# ── File Upload ───────────────────────────────────────────────────────────────
section_header("02 · AUDIO INPUT")

uploaded = st.file_uploader(
    "Drop audio file here — .wav / .flac / .mp3",
    type=['wav', 'flac', 'mp3'],
    label_visibility="collapsed"
)

# File validation
if uploaded is not None:
    file_size = len(uploaded.getvalue())
    file_size_mb = file_size / (1024 * 1024)
    
    # Check file size (max 50MB)
    if file_size_mb > 50:
        st.error(f"⚠️ File too large: {file_size_mb:.1f}MB. Maximum size is 50MB.")
        uploaded = None
    # Check minimum file size (at least 1KB)
    elif file_size < 1024:
        st.error("⚠️ File too small or corrupted. Please upload a valid audio file.")
        uploaded = None
    else:
        # Show file info
        st.markdown(f"""
        <div style="background:#0E1220; border:1px solid #1E2840; border-radius:10px;
                    padding:0.75rem 1rem; margin:0.5rem 0; font-size:0.85rem;">
            <span style="color:#4A5568;">📎</span>
            <span style="color:#8899AA; margin-left:0.5rem;">{uploaded.name}</span>
            <span style="color:#4A5568; margin-left:1rem;">({file_size_mb:.2f} MB)</span>
        </div>
        """, unsafe_allow_html=True)


# ── Helper: Robust Audio Loading ─────────────────────────────────────────────
def load_audio_robust(filepath):
    """Load audio with multiple fallback methods"""
    try:
        # Method 1: librosa with audioread backend
        audio, sr = librosa.load(filepath, sr=None)
        return audio, sr
    except Exception as e1:
        try:
            # Method 2: librosa with soundfile backend
            import soundfile as sf
            audio, sr = sf.read(filepath)
            if len(audio.shape) > 1:  # stereo to mono
                audio = audio.mean(axis=1)
            return audio, sr
        except Exception as e2:
            try:
                # Method 3: scipy.io.wavfile
                from scipy.io import wavfile
                sr, audio = wavfile.read(filepath)
                # Convert to float32 and normalize
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                elif audio.dtype == np.int32:
                    audio = audio.astype(np.float32) / 2147483648.0
                # stereo to mono
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                return audio, sr
            except Exception as e3:
                # All methods failed
                st.error(f"""
                ⚠️ **Audio Loading Failed**
                
                Could not read the audio file. Possible causes:
                - Corrupted or incomplete file
                - Unsupported codec or container format
                - File extension mismatch
                
                **Try:**
                - Re-export the audio as WAV (PCM 16-bit)
                - Use a different audio file
                - Convert using: `ffmpeg -i input.mp3 output.wav`
                
                **Technical details:**
                - Method 1 (librosa): {str(e1)[:100]}
                - Method 2 (soundfile): {str(e2)[:100]}
                - Method 3 (scipy): {str(e3)[:100]}
                """)
                return None, None


# ── Analysis ──────────────────────────────────────────────────────────────────
if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    # ── Audio Player ──
    section_header("03 · WAVEFORM ANALYSIS")
    st.audio(uploaded)

    # Load audio with robust error handling
    audio_raw, sr = load_audio_robust(tmp_path)
    
    if audio_raw is None:
        # Error message already shown in load_audio_robust
        os.unlink(tmp_path)
        st.stop()
    
    duration = len(audio_raw) / sr

    # File metadata row
    st.markdown(f"""
    <div style="display:flex; gap:0.75rem; margin:0.75rem 0 1rem;">
        <div class="metric-chip">
            <div class="metric-chip-label">Duration</div>
            <div class="metric-chip-value">{duration:.2f}s</div>
        </div>
        <div class="metric-chip">
            <div class="metric-chip-label">Sample Rate</div>
            <div class="metric-chip-value">{sr // 1000}kHz</div>
        </div>
        <div class="metric-chip">
            <div class="metric-chip-label">Samples</div>
            <div class="metric-chip-value">{len(audio_raw) // 1000}K</div>
        </div>
        <div class="metric-chip">
            <div class="metric-chip-label">File</div>
            <div class="metric-chip-value" style="font-size:0.8rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{uploaded.name}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Visualization ──
    try:
        plt.rcParams.update({
            'figure.facecolor':  '#0A0D16',
            'axes.facecolor':    '#0A0D16',
            'axes.edgecolor':    '#1E2840',
            'axes.labelcolor':   '#4A5568',
            'xtick.color':       '#2A3448',
            'ytick.color':       '#2A3448',
            'text.color':        '#8899AA',
            'grid.color':        '#1E2840',
            'grid.linestyle':    '--',
            'grid.linewidth':    0.5,
        })

        fig, axes = plt.subplots(1, 2, figsize=(12, 3.2), facecolor='#0A0D16')
        fig.patch.set_facecolor('#0A0D16')
        fig.subplots_adjust(hspace=0, wspace=0.35)

        # Waveform
        t = np.linspace(0, duration, len(audio_raw))
        axes[0].plot(t, audio_raw, lw=0.6, color='#00E5CC', alpha=0.85)
        axes[0].fill_between(t, audio_raw, 0, alpha=0.08, color='#00E5CC')
        axes[0].set_xlim(0, duration)
        axes[0].set_xlabel('Time (s)', fontsize=9, labelpad=6)
        axes[0].set_ylabel('Amplitude', fontsize=9, labelpad=6)
        axes[0].set_title('Waveform', fontsize=10, fontweight='600',
                          color='#CBD5E0', pad=10, loc='left')
        axes[0].grid(True, alpha=0.3)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        # Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=audio_raw, sr=sr, n_mels=64, fmax=7000)
        img = librosa.display.specshow(
            librosa.power_to_db(mel, ref=np.max),
            sr=sr, x_axis='time', y_axis='mel',
            fmax=7000, ax=axes[1], cmap='inferno'
        )
        axes[1].set_title('Mel Spectrogram · Wideband', fontsize=10, fontweight='600',
                          color='#CBD5E0', pad=10, loc='left')
        axes[1].set_xlabel('Time (s)', fontsize=9, labelpad=6)
        axes[1].set_ylabel('Mel Freq (Hz)', fontsize=9, labelpad=6)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        cbar = fig.colorbar(img, ax=axes[1], format='%+2.0f dB', pad=0.02)
        cbar.ax.yaxis.set_tick_params(color='#2A3448', labelsize=8)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#4A5568')
        cbar.outline.set_edgecolor('#1E2840')

        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.warning(f"⚠️ Could not generate waveform visualization: {str(e)}")
        st.info("Continuing with audio analysis...")

    # ── Analyze Button ────────────────────────────────────────────────────────
    section_header("04 · FORENSIC ANALYSIS")

    if st.button("⬡  RUN VOICE FORENSICS", type="primary"):
        with st.spinner("Simulating VoIP channel & running neural analysis..."):
            # Fake progress for dramatic effect
            progress_bar = st.progress(0, text="Pre-processing audio signal...")
            
            try:
                for pct in range(0, 60, 15):
                    time.sleep(0.18)
                    progress_bar.progress(pct, text=f"Simulating Opus codec artifacts... {pct}%")
                
                # Preprocessing step with error handling
                try:
                    tensor = preprocess_for_inference(tmp_path, bitrate_mode=bm, snr_db=25.0)
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"""
                    ⚠️ **Preprocessing Failed**
                    
                    Error during audio preprocessing: {str(e)}
                    
                    **Try:**
                    - Use a different audio file
                    - Ensure audio is not corrupted
                    - Convert to WAV format: `ffmpeg -i input.mp3 -ar 16000 output.wav`
                    """)
                    st.stop()
                
                progress_bar.progress(70, text="Running RawNet3 inference...")
                
                # Inference step with error handling
                try:
                    result = predict(model, tensor, device)
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"""
                    ⚠️ **Model Inference Failed**
                    
                    Error during neural network inference: {str(e)}
                    
                    This is likely a model or tensor shape issue.
                    """)
                    st.stop()
                
                progress_bar.progress(100, text="Analysis complete.")
                time.sleep(0.3)
                progress_bar.empty()
                
            except Exception as e:
                progress_bar.empty()
                st.error(f"⚠️ **Unexpected Error:** {str(e)}")
                st.stop()

        # ── Verdict Card ──
        is_bonafide = result['label'] == 'bonafide'
        card_class  = "bonafide" if is_bonafide else "spoofed"
        verdict_txt = "✓ BONAFIDE" if is_bonafide else "✕ SPOOFED"
        verdict_sub = "Voice signature appears genuine" if is_bonafide else "Synthetic / cloned voice detected"
        conf_pct    = f"{result['confidence']:.1%}"

        st.markdown(f"""
        <div class="result-card {card_class}">
            <div class="result-glow-{'b' if is_bonafide else 's'}"></div>
            <div class="result-verdict {card_class}">{verdict_txt}</div>
            <div class="result-desc">{verdict_sub}</div>
            <div class="result-confidence">
                <div class="confidence-num {card_class}">{conf_pct}</div>
                <div class="confidence-label">confidence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability Bars ──
        pb = float(result['prob_bonafide'])
        ps = float(result['prob_spoofed'])

        # Normalize if model returns 0-100 instead of 0-1
        if pb > 1.0 or ps > 1.0:
            pb /= 100.0
            ps /= 100.0
        pb = max(0.0, min(1.0, pb))
        ps = max(0.0, min(1.0, ps))

        # Pre-format all values as plain strings — no f-string with CSS braces
        pb_str    = "{:.4f}".format(pb)
        ps_str    = "{:.4f}".format(ps)
        pb_width  = "{:.2f}".format(max(1.0, pb * 100) if pb > 0 else 0)
        ps_width  = "{:.2f}".format(max(1.0, ps * 100) if ps > 0 else 0)

        # Build HTML via string concat (avoids f-string + CSS-brace conflicts)
        prob_html = (
            '<div style="background:#0E1220;border:1px solid #1E2840;border-radius:14px;'
            'padding:1.25rem 1.5rem;margin-top:1rem;">'
            '<div style="font-size:0.72rem;font-family:monospace;color:#4A5568;'
            'text-transform:uppercase;letter-spacing:0.1em;margin-bottom:1rem;">'
            'Probability Distribution</div>'
            # BONAFIDE row
            '<div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem;">'
            '<div style="font-family:monospace;font-size:0.75rem;color:#8899AA;width:90px;flex-shrink:0;">BONAFIDE</div>'
            '<div style="flex:1;height:8px;background:#0A0D16;border-radius:4px;overflow:hidden;">'
            '<div style="height:100%;border-radius:4px;background:linear-gradient(90deg,#00A86B,#00E87A);width:'
            + pb_width + '%;"></div></div>'
            '<div style="font-family:monospace;font-size:0.82rem;font-weight:700;width:52px;text-align:right;color:#00E87A;">'
            + pb_str + '</div></div>'
            # SPOOFED row
            '<div style="display:flex;align-items:center;gap:1rem;">'
            '<div style="font-family:monospace;font-size:0.75rem;color:#8899AA;width:90px;flex-shrink:0;">SPOOFED</div>'
            '<div style="flex:1;height:8px;background:#0A0D16;border-radius:4px;overflow:hidden;">'
            '<div style="height:100%;border-radius:4px;background:linear-gradient(90deg,#C0203A,#FF4D6D);width:'
            + ps_width + '%;"></div></div>'
            '<div style="font-family:monospace;font-size:0.82rem;font-weight:700;width:52px;text-align:right;color:#FF4D6D;">'
            + ps_str + '</div></div>'
            '</div>'
        )
        st.markdown(prob_html, unsafe_allow_html=True)

        # ── Extra Metrics ──
        margin    = abs(pb - ps)
        certainty = "High" if margin > 0.6 else ("Moderate" if margin > 0.3 else "Low")
        m_str     = "{:.4f}".format(margin)

        chip = (
            '<div style="display:flex;gap:0.75rem;margin-top:1.25rem;">'
            + ''.join([
                '<div style="flex:1;background:#0A0D16;border:1px solid #1E2840;border-radius:10px;'
                'padding:0.75rem 1rem;text-align:center;">'
                '<div style="font-size:0.68rem;color:#4A5568;text-transform:uppercase;letter-spacing:0.1em;'
                'font-family:monospace;margin-bottom:0.3rem;">' + label + '</div>'
                '<div style="font-size:1.05rem;font-weight:600;font-family:monospace;color:' + color + ';">' + val + '</div>'
                '</div>'
                for label, val, color in [
                    ("P(Bonafide)", pb_str, "#00E87A"),
                    ("P(Spoofed)",  ps_str, "#FF4D6D"),
                    ("Margin",      m_str,  "#CBD5E0"),
                    ("Certainty",   certainty, "#CBD5E0"),
                ]
            ])
            + '</div>'
        )
        st.markdown(chip, unsafe_allow_html=True)

    os.unlink(tmp_path)

else:
    # ── Empty State ──
    st.markdown("""
    <div style="text-align:center; padding:3rem 1rem; border:1px dashed #1E2840;
                border-radius:16px; margin:1.5rem 0; background:#080B14;">
        <div style="font-size:2.5rem; margin-bottom:0.75rem; opacity:0.4;">🎙️</div>
        <div style="color:#4A5568; font-size:0.9rem; line-height:1.6;">
            Upload a .wav, .flac, or .mp3 file<br>
            <span style="font-size:0.8rem; color:#2A3448;">Supported: mono/stereo · any sample rate · max 60s recommended</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    MODEL: RAWNET3-GATEDCONVNEXT &nbsp;·&nbsp;
    DATASET: ASVSPOOF 2021 DF &nbsp;·&nbsp;
    PIPELINE: VOIP/WHATSAPP (OPUS WB) &nbsp;·&nbsp;
    VoiceGuard v2.1
</div>
""", unsafe_allow_html=True)