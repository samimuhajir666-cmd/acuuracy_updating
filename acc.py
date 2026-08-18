import io
import os
import tempfile
import numpy as np
import requests
import scipy.io.wavfile as wav
import scipy.signal as signal
import streamlit as st
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
from unidecode import unidecode

# Optional Local Whisper Import
try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False

# ============================
# 🖥️ PAGE CONFIG & ENVS
# ============================
st.set_page_config(
    page_title="Multi-LLM Voice AI Agent",
    page_icon="🤖",
    layout="wide"
)

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY") or st.secrets.get("DEEPGRAM_API_KEY", None)
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)

TECHNICAL_VOCAB = [
    "Python", "Streamlit", "FastAPI", "Django", "JavaScript", 
    "Deepgram", "Groq", "Whisper", "OpenAI", "POS", "API",
    "Machine Learning", "System Error", "Troubleshoot"
]

SYSTEM_PROMPT = (
    "Aap ek highly intelligent technical support AI agent hain. "
    "User ke masle ka jawab concise, clear, aur polite Roman Urdu / English mixed format mein dein."
)

# ============================
# 🎚️ AUDIO PRE-PROCESSING
# ============================
def clean_audio_signal(audio_bytes):
    """Bandpass filter aur normalization se voice clear karna."""
    try:
        audio_file = io.BytesIO(audio_bytes)
        sample_rate, audio_data = wav.read(audio_file)

        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        audio_data = audio_data.astype(np.float64)
        
        # Bandpass filter (70Hz - 7600Hz)
        nyquist = 0.5 * sample_rate
        low = max(0.001, 70 / nyquist)
        high = min(0.99, 7600 / nyquist)
        b, a = signal.butter(3, [low, high], btype="band")
        filtered = signal.filtfilt(b, a, audio_data)

        # Peak Normalization
        max_val = np.max(np.abs(filtered))
        if max_val > 1e-8:
            filtered = (filtered / max_val) * 32767.0

        processed = np.clip(filtered, -32768, 32767).astype(np.int16)
        output_buffer = io.BytesIO()
        wav.write(output_buffer, sample_rate, processed)
        output_buffer.seek(0)
        return output_buffer.read()
    except Exception:
        return audio_bytes

# ============================
# 🎙️ STT ENGINES (Transcribers)
# ============================

# 1. Deepgram Engine
def transcribe_deepgram(audio_bytes):
    if not DEEPGRAM_API_KEY:
        raise ValueError("Deepgram API Key missing hai! .env check karein.")

    params = [
        ("model", "nova-3"),
        ("language", "multi"),
        ("smart_format", "true"),
        ("punctuate", "true"),
    ]
    for term in TECHNICAL_VOCAB:
        params.append(("keyterm", term))

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav",
    }
    res = requests.post("https://api.deepgram.com/v1/listen", params=params, headers=headers, data=audio_bytes, timeout=30)
    if res.status_code != 200:
        raise RuntimeError(f"Deepgram Error: {res.text[:200]}")

    data = res.json()
    channels = data.get("results", {}).get("channels", [])
    if channels and channels[0].get("alternatives"):
        alt = channels[0]["alternatives"][0]
        return unidecode(alt.get("transcript", "")), float(alt.get("confidence", 0.0))
    return "", 0.0

# 2. Faster-Whisper Engine (Local)
@st.cache_resource(show_spinner="Local Whisper Load Ho Raha Hai...")
def get_whisper_model():
    if not HAS_FASTER_WHISPER:
        return None
    return WhisperModel("base", device="cpu", compute_type="int8")

def transcribe_whisper(audio_bytes):
    model = get_whisper_model()
    if not model:
        raise RuntimeError("faster-whisper package missing hai!")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(
            tmp_path,
            beam_size=8,
            temperature=0.0,
            vad_filter=True,
            initial_prompt="Keywords: " + ", ".join(TECHNICAL_VOCAB)
        )
        text = " ".join([s.text.strip() for s in segments]).strip()
        return unidecode(text), float(info.language_probability)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ============================
# 🤖 MULTI-LLM ENGINES
# ============================

# LLM 1: Groq (Llama 3.3 70B - Ultra Fast)
def get_groq_response(user_text):
    if not GROQ_API_KEY:
        return "❌ GROQ_API_KEY missing hai!"
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text}
        ],
        "temperature": 0.3
    }
    res = requests.post(url, json=payload, headers=headers, timeout=20)
    if res.status_code == 200:
        return res.json()["choices"][0]["message"]["content"]
    return f"Groq Error ({res.status_code}): {res.text[:150]}"

# LLM 2: OpenAI (GPT-4o Mini / GPT-4o)
def get_openai_response(user_text):
    if not OPENAI_API_KEY:
        return "❌ OPENAI_API_KEY missing hai!"

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o-mini",  # Highly accurate & cost efficient
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text}
        ],
        "temperature": 0.3
    }
    res = requests.post(url, json=payload, headers=headers, timeout=20)
    if res.status_code == 200:
        return res.json()["choices"][0]["message"]["content"]
    return f"OpenAI Error ({res.status_code}): {res.text[:150]}"

# ============================
# 🖥️ STREAMLIT UI
# ============================
def main():
    st.title("🤖 Multi-Model AI Voice Agent")
    st.caption("Select your preferred Speech-to-Text Engine and LLM Intelligence below.")

    # Sidebar Controls
    st.sidebar.header("⚙️ Model Pipeline Selection")

    # STT Model Switcher
    stt_choice = st.sidebar.selectbox(
        "1. Select Speech-to-Text Model:",
        ["⚡ Deepgram Nova-3 (Cloud / Recommended)", "💻 Faster-Whisper (Local CPU)"]
    )

    # LLM Model Switcher
    llm_choice = st.sidebar.selectbox(
        "2. Select LLM Agent Brain:",
        ["🔥 Groq (Llama-3.3 70B - Ultra Fast)", "🧠 OpenAI (GPT-4o Mini)"]
    )

    enable_filter = st.sidebar.checkbox("✨ Noise & Audio Filter", value=True)

    st.markdown("---")
    st.subheader("🎤 Speak to Agent")

    audio = mic_recorder(
        start_prompt="🔴 Record Voice",
        stop_prompt="⬛ Stop Recording",
        key="voice_rec"
    )

    if audio and audio.get("bytes"):
        raw_bytes = audio["bytes"]
        st.audio(raw_bytes, format="audio/wav")

        if st.button("🚀 Process & Get Agent Jawab"):
            with st.spinner("Processing Voice & Pipeline..."):
                cleaned_bytes = clean_audio_signal(raw_bytes) if enable_filter else raw_bytes

                try:
                    # 1. Transcribe Step
                    if "Deepgram" in stt_choice:
                        transcript, conf = transcribe_deepgram(cleaned_bytes)
                        engine_used = "Deepgram Nova-3"
                    else:
                        transcript, conf = transcribe_whisper(cleaned_bytes)
                        engine_used = "Faster-Whisper (Local)"

                    st.success(f"✅ Transcribed using **{engine_used}**")
                    st.info(f"**User Said:** {transcript if transcript else 'No speech detected.'}")
                    st.caption(f"Confidence: {conf:.2f}")

                    # 2. LLM Processing Step
                    if transcript:
                        with st.spinner("LLM Processing..."):
                            if "Groq" in llm_choice:
                                response = get_groq_response(transcript)
                                brain_used = "Groq Llama-3.3"
                            else:
                                response = get_openai_response(transcript)
                                brain_used = "OpenAI GPT-4o Mini"

                            st.markdown(f"### 🤖 Agent Response (*Powered by {brain_used}*):")
                            st.success(response)

                except Exception as e:
                    st.error(f"❌ Error in Pipeline: {str(e)}")

if __name__ == "__main__":
    main()
