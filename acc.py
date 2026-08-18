import os
import json
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
import noisereduce as nr
import streamlit as st
from pathlib import Path
from faster_whisper import WhisperModel

# ==============================================================================
# 🛠️ AUDIO CLEANUP (DSP FILTERS - NOISE CANCELLATION & SHOUTING NORMALIZER)
# ==============================================================================
def clean_audio_signal(audio_path):
    """Applies ButterWorth, Noise Reduction, and AGC to raw audio before AI processing."""
    sample_rate, data = wav.read(audio_path)
    
    # 1. Convert to float32
    if data.dtype == np.int16:
        audio_float = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio_float = data.astype(np.float32) / 2147483648.0
    else:
        audio_float = data.astype(np.float32)
        
    if len(audio_float.shape) > 1:
        audio_float = np.mean(audio_float, axis=1)
        
    # 2. Automatic Gain Control (Normalize shouting / loud peaks)
    max_peak = np.max(np.abs(audio_float))
    if max_peak > 0.70:
        audio_float = np.tanh(audio_float / max_peak) * 0.70
        
    # 3. Butterworth Bandpass Filter (85Hz - 3800Hz)
    nyquist = 0.5 * sample_rate
    low_cutoff = 85.0 / nyquist
    high_cutoff = min(3800.0 / nyquist, 0.99)
    b, a = signal.butter(4, [low_cutoff, high_cutoff], btype="band")
    filtered_audio = signal.filtfilt(b, a, audio_float)
    
    # 4. Adaptive Background Noise Cancellation
    reduced_noise = nr.reduce_noise(
        y=filtered_audio, 
        sr=sample_rate, 
        prop_decrease=0.85,
        n_fft=1024
    )
    
    # Save processed PCM WAV
    clean_signal = np.clip(reduced_noise * 32768.0, -32768, 32767).astype(np.int16)
    cleaned_path = audio_path.parent / f"cleaned_{audio_path.name}"
    wav.write(cleaned_path, sample_rate, clean_signal)
    return cleaned_path

# ==============================================================================
# 🤖 WHISPER AI INITIALIZATION
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_whisper_model():
    return WhisperModel("base", device="cpu", compute_type="int8")

# ==============================================================================
# 🖥️ STREAMLIT APP LAYOUT
# ==============================================================================
st.set_page_config(page_title="Voice to Text AI", page_icon="🎙️", layout="centered")

st.title("🎙️ Live Voice Recording & Transcription")
st.write("Apni aawaz record karein — Noise clean karke auto-transcribe hojayega.")

# Streamlit Native Live Microphone Input
audio_recording = st.audio_input("Microphone se record karein")

if audio_recording is not None:
    st.audio(audio_recording)
    
    if st.button("🚀 Transcribe Recording", type="primary"):
        temp_dir = Path("temp_workspace")
        temp_dir.mkdir(exist_ok=True)
        
        raw_audio_path = temp_dir / "live_recording.wav"
        
        # Save audio byte stream to WAV file
        with open(raw_audio_path, "wb") as f:
            f.write(audio_recording.getbuffer())
            
        try:
            # Step 1: Clean background noise & balance shouting volume
            with st.spinner("🧼 Noise cancel aur audio balance ho raha hai..."):
                cleaned_audio_path = clean_audio_signal(raw_audio_path)
                
            # Step 2: Transcribe using Whisper Model
            with st.spinner("⚡ AI Text mein convert kar raha hai..."):
                model = load_whisper_model()
                segments, info = model.transcribe(
                    str(cleaned_audio_path),
                    vad_filter=True,
                    beam_size=5
                )
                
                full_text = " ".join([segment.text for segment in segments])
                
            st.success("✅ Transcription Complete!")
            
            # Step 3: Show result
            st.subheader("📝 Transcribed Text:")
            st.text_area("Result", value=full_text, height=200)
            
            # Download button
            st.download_button(
                label="📥 Download Text File",
                data=full_text,
                file_name="recording_transcript.txt",
                mime="text/plain"
            )
            
            # Temporary files cleanup
            if raw_audio_path.exists():
                os.remove(raw_audio_path)
            if cleaned_audio_path.exists():
                os.remove(cleaned_audio_path)

        except Exception as err:
            st.error(f"❌ Error aaya: {err}")
