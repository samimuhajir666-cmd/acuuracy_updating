import io
import os
import re
import html
import requests
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import noisereduce as nr
import streamlit as st
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
from unidecode import unidecode


# ============================
# 🖥️ STREAMLIT PAGE CONFIG
# ============================
st.set_page_config(
    page_title="AI Speech-to-Text - Deepgram Nova-3",
    page_icon="🎤",
    layout="centered",
)

load_dotenv()

# ============================
# 🔑 API KEYS SETUP
# ============================
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    try:
        DEEPGRAM_API_KEY = st.secrets.get("DEEPGRAM_API_KEY")
    except Exception:
        DEEPGRAM_API_KEY = None

if not DEEPGRAM_API_KEY:
    st.error("DEEPGRAM_API_KEY not found. Put it in .env or Streamlit Secrets.")
    st.stop()


# ============================
# 🎙️ DEEPGRAM CONFIGURATION
# ============================
DEEPGRAM_API_URL = "https://api.deepgram.com/v1/listen"
DEEPGRAM_MODEL = "nova-3"
DEEPGRAM_LANGUAGE = "multi"
DEEPGRAM_TIMEOUT = 60

DEEPGRAM_KEYTERMS = [
    "Python", "Streamlit", "Jupyter", "Matplotlib", "Plotly",
    "NumPy", "SciPy", "Deepgram", "AI", "machine learning",
    "deep learning", "API", "API key", "variable", "function",
    "class", "list", "dictionary", "tuple", "integer", "string",
    "float", "Flask", "FastAPI", "JavaScript", "HTML", "CSS",
]

def force_roman_script(text):
    if not text:
        return text
    has_non_ascii = bool(re.search(r'[^\x00-\x7F]', text))
    if not has_non_ascii:
        return text
    return unidecode(text)

# ============================
# 🎙️ DEEPGRAM TRANSCRIBE
# ============================
def transcribe_with_deepgram(processed_bytes, debug=False):
    params = [
        ("model", DEEPGRAM_MODEL),
        ("language", DEEPGRAM_LANGUAGE),
        ("smart_format", "true"),
        ("punctuate", "true"),
        ("utterances", "true"),
        ("numerals", "true"),
    ]

    for term in DEEPGRAM_KEYTERMS:
        params.append(("keyterm", term))

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav",
    }

    try:
        response = requests.post(
            DEEPGRAM_API_URL,
            params=params,
            headers=headers,
            data=processed_bytes,
            timeout=DEEPGRAM_TIMEOUT,
        )
    except requests.RequestException as e:
        if debug:
            st.exception(e)
        raise RuntimeError(f"Could not reach Deepgram: {e}") from e

    if response.status_code != 200:
        detail = response.text[:1200]
        raise RuntimeError(f"Deepgram API error {response.status_code}: {detail}")

    try:
        data = response.json()
    except Exception as e:
        raise RuntimeError("Deepgram returned invalid JSON.") from e

    results = data.get("results", {})
    channels = results.get("channels", [])

    if not channels:
        return {"text": "", "confidence": 0.0, "raw": data}

    alternatives = channels[0].get("alternatives", [])
    if not alternatives:
        return {"text": "", "confidence": 0.0, "raw": data}

    alternative = alternatives[0]
    transcript = (alternative.get("transcript") or "").strip()
    confidence = float(alternative.get("confidence", 0.0) or 0.0)

    return {
        "text": force_roman_script(transcript),
        "confidence": confidence,
        "raw": data,
    }

# ============================
# 🎚️ AUDIO PROCESSING HELPERS
# ============================
MIN_RMS_ENERGY = 35.0
MIN_DURATION_SECONDS = 0.45
MAX_DURATION_SECONDS = 120
VAD_FRAME_MS = 30
MIN_SPEECH_SECONDS = 0.20
NOISE_FLOOR_PERCENTILE = 10
SPEECH_ABOVE_NOISE_FACTOR = 2.0

def frame_energies(audio_data, sample_rate, frame_ms=VAD_FRAME_MS):
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    energies = []
    for start in range(0, len(audio_data), frame_len):
        chunk = audio_data[start:start + frame_len]
        if len(chunk) == 0:
            continue
        energies.append(float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2))))
    return energies

def contains_real_speech(audio_data, sample_rate):
    if audio_data is None or len(audio_data) == 0:
        return False
    energies = frame_energies(audio_data, sample_rate)
    if not energies:
        return False
    noise_floor = np.percentile(energies, NOISE_FLOOR_PERCENTILE)
    dynamic_threshold = max(noise_floor * SPEECH_ABOVE_NOISE_FACTOR, MIN_RMS_ENERGY)
    speech_frame_count = sum(1 for energy in energies if energy > dynamic_threshold)
    speech_seconds = speech_frame_count * VAD_FRAME_MS / 1000.0
    return speech_seconds >= MIN_SPEECH_SECONDS

def process_audio_buffer(audio_bytes, debug=False):
    try:
        audio_file = io.BytesIO(audio_bytes)
        sample_rate, audio_data = wav.read(audio_file)

        if sample_rate <= 0:
            return None

        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        audio_data = audio_data.astype(np.float64)
        duration = len(audio_data) / float(sample_rate)

        if duration < MIN_DURATION_SECONDS or duration > MAX_DURATION_SECONDS:
            return None

        if not contains_real_speech(audio_data, sample_rate):
            return None

        audio_data = np.clip(audio_data, -32768, 32767).astype(np.int16)
        output_buffer = io.BytesIO()
        wav.write(output_buffer, sample_rate, audio_data)
        output_buffer.seek(0)

        return {
            "processed_bytes": output_buffer.read(),
            "sample_rate": int(sample_rate),
            "duration": float(duration),
        }
    except Exception as e:
        if debug:
            st.exception(e)
        return None

# ============================
# 🧠 SESSION STATE
# ============================
if "last_transcription" not in st.session_state:
    st.session_state.last_transcription = ""

# ============================
# 🧩 UI DESIGN & LAYOUT
# ============================
st.title("🎤 Deepgram Nova-3 Speech-to-Text")
st.caption("Powered by Deepgram Nova-3 Multi-Language Model")
st.info("Record your voice below to instantly transcribe it into text.")

# Microphone Widget
audio_output = mic_recorder(
    start_prompt="🔴 Record Voice",
    stop_prompt="⏹️ Stop Recording",
    just_once=True,
    use_container_width=True,
    format="wav",
    key="listener_mic",
)

if audio_output:
    audio_bytes = audio_output.get("bytes")

    if audio_bytes:
        with st.spinner("⏳ Processing audio buffer..."):
            result = process_audio_buffer(audio_bytes)

        if result is None:
            st.warning("⚠️ Recording was too quiet or short. Please speak clearly and try again.")
        else:
            processed_bytes = result["processed_bytes"]

            # Deepgram Transcription
            with st.spinner("⚡ Transcribing with Deepgram Nova-3..."):
                try:
                    transcription_result = transcribe_with_deepgram(processed_bytes)
                    text_from_voice = transcription_result["text"].strip()
                    confidence = float(transcription_result["confidence"])

                    if text_from_voice:
                        st.session_state.last_transcription = text_from_voice
                        st.success("✅ Transcribed successfully using Deepgram Nova-3")
                        st.markdown(f"**Transcription:** {text_from_voice}")
                        st.caption(f"Confidence: {confidence:.2f}")
                    else:
                        st.warning("⚠️ No recognizable speech found in the audio.")
                except Exception as e:
                    st.error(f"❌ Transcription error: {e}")

# Clear & Reset Controls
st.divider()
if st.button("🗑️ Clear & Reset", use_container_width=True):
    st.session_state.last_transcription = ""
    st.rerun()
