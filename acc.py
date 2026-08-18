import io
import os
import json
import numpy as np
import scipy.signal as signal
import noisereduce as nr
import streamlit as st
from pathlib import Path
from datetime import datetime
from faster_whisper import WhisperModel

# ==============================================================================
# 🛠️ TRANSCRIPTION SERVICE CLASS (ENGINE WITH BACKGROUND & LOUDNESS FIXES)
# ==============================================================================
class TranscriptionService:
    """Production-ready transcription service using faster-whisper with DSP overrides."""
    
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        """Initialize the transcription service."""
        st.write(f"🔄 Waking up AI Engine: `{model_size}` on `{device}` ({compute_type})...")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
        st.write("✅ AI Brain loaded successfully!")
    
    def apply_advanced_acoustic_cleanup(self, audio_path):
        """
        Applies mathematical filters to stabilize loud voices (shouting) 
        and eliminate background machinery/room noise floors before AI parsing.
        """
        import scipy.io.wavfile as wav
        
        # Load the raw audio data array safely
        sample_rate, data = wav.read(audio_path)
        
        # 1. Convert to floating point representation for precise matrix calculations
        if data.dtype == np.int16:
            audio_float = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            audio_float = data.astype(np.float32) / 2147483648.0
        else:
            audio_float = data.astype(np.float32)
            
        # Handle multi-channel stereo down to uniform mono
        if len(audio_float.shape) > 1:
            audio_float = np.mean(audio_float, axis=1)
            
        # 2. AUTOMATIC GAIN CONTROL (AGC) & LIMITER FOR LOUD SHOUTING VOICES
        # If someone speaks too loudly, this squashes the clipping peaks smoothly
        max_peak = np.max(np.abs(audio_float))
        if max_peak > 0.70:
            # Apply soft-knee compression scaling matrix to normalize loud volumes
            audio_float = np.tanh(audio_float / max_peak) * 0.70
            
        # 3. BUTTERWORTH BANDPASS FILTER (Clears low hums and high static hiss)
        nyquist = 0.5 * sample_rate
        low_cutoff = 85.0 / nyquist
        high_cutoff = min(3800.0 / nyquist, 0.99)
        b, a = signal.butter(4, [low_cutoff, high_cutoff], btype="band")
        filtered_audio = signal.filtfilt(b, a, audio_float)
        
        # 4. ADAPTIVE BACKGROUND NOISE CANCELER (Removes background fan/traffic noise)
        reduced_noise = nr.reduce_noise(
            y=filtered_audio, 
            sr=sample_rate, 
            prop_decrease=0.85, # Drop noise signature by 85%
            n_fft=1024
        )
        
        # Convert back safely to standard production 16-bit PCM integer WAV formatting
        clean_signal = np.clip(reduced_noise * 32768.0, -32768, 32767).astype(np.int16)
        
        # Overwrite the temporary file with our pristine, filtered audio track
        cleaned_path = audio_path.parent / f"cleaned_{audio_path.name}"
        wav.write(cleaned_path, sample_rate, clean_signal)
        return cleaned_path
    
    def transcribe_file(self, audio_path, output_format="txt", **kwargs):
        """
        Transcribe an audio file with custom acoustic preprocessing hooks.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        # Execute background filtering and loud-speaker stabilization layers first
        with st.spinner("🧼 Cleaning background noise and balancing loud vocal peaks..."):
            cleaned_audio_target = self.apply_advanced_acoustic_cleanup(audio_path)
        
        # Transcribe clean audio tracking parameters
        segments, info = self.model.transcribe(
            str(cleaned_audio_target),
            word_timestamps=True,
            **kwargs
        )
        
        # Collect results data layout
        result = {
            "file": str(audio_path.name),
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "segments": []
        }
        
        full_text_parts = []
        for segment in segments:
            segment_data = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "words": [
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability
                    }
                    for word in segment.words
                ]
            }
            result["segments"].append(segment_data)
            full_text_parts.append(segment.text)
        
        result["text"] = " ".join(full_text_parts)
        
        # Generate export paths safely inside temporary workspace
        output_path = audio_path.parent / f"{audio_path.stem}_transcript"
        
        if output_format == "txt":
            self._save_txt(result, output_path.with_suffix(".txt"))
            final_file = output_path.with_suffix(".txt")
        elif output_format == "json":
            self._save_json(result, output_path.with_suffix(".json"))
            final_file = output_path.with_suffix(".json")
        elif output_format == "srt":
            self._save_srt(result, output_path.with_suffix(".srt"))
            final_file = output_path.with_suffix(".srt")
        elif output_format == "vtt":
            self._save_vtt(result, output_path.with_suffix(".vtt"))
            final_file = output_path.with_suffix(".vtt")
            
        # Clean up temporary processing step files to preserve host disk space
        if cleaned_audio_target.exists():
            os.remove(cleaned_audio_target)
            
        return result, final_file

    def _save_txt(self, result, path):
        """Save as plain text."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(result["text"])
    
    def _save_json(self, result, path):
        """Save as JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    def _save_srt(self, result, path):
        """Save as SRT subtitles."""
        with open(path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(result["segments"], start=1):
                start = self._format_srt_time(seg["start"])
                end = self._format_srt_time(seg["end"])
                f.write(f"{i}\n{start} --> {end}\n{seg['text']}\n\n")
    
    def _save_vtt(self, result, path):
        """Save as WebVTT."""
        with open(path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for seg in result["segments"]:
                start = self._format_vtt_time(seg["start"])
                end = self._format_vtt_time(seg["end"])
                f.write(f"{start} --> {end}\n{seg['text']}\n\n")
    
    def _format_srt_time(self, seconds):
        """Format time for SRT."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _format_vtt_time(self, seconds):
        """Format time for VTT."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

# ==============================================================================
# 🖥️ STREAMLIT WEB INTERFACE LAYOUT (DEPLOYMENT READY)
# ==============================================================================
st.set_page_config(page_title="Say to Words Web AI", page_icon="🤖", layout="wide")
st.title("🤖 Say to Words - Web AI Transcription Agent")
st.caption("Production Build: Audio Normalization Engine Enabled (Anti-Shouting + Background Cancellation Filter)")

# Sidebar configurations controls panel
st.sidebar.header("⚙️ Model Architecture Settings")
model_size = st.sidebar.selectbox("Whisper Model Scale", ["tiny", "base", "small", "medium"], index=1)
device_choice = st.sidebar.selectbox("Execution Hardware", ["cpu", "cuda"], index=0)
compute_choice = st.sidebar.selectbox("Quantization Compute Type", ["int8", "float16"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("🎯 Transcription Tuning")
beam_size_val = st.sidebar.slider("Beam Size Accuracy", 1, 10, 5)
use_vad = st.sidebar.checkbox("Enable Voice Activity Detection (VAD Filter)", value=True)
output_ext = st.sidebar.selectbox("Export File Target Format", ["txt", "json", "srt", "vtt"], index=0)

# Main container file upload dropzone
uploaded_file = st.file_uploader("Upload your Audio File (MP3, WAV, M4A, etc.)", type=["mp3", "wav", "m4a", "ogg"])

if uploaded_file is not None:
    st.info(f"📁 Target Received: `{uploaded_file.name}`. Initializing Acoustic Matrix filters...")
    
    # FIXED: Indentation block corrected perfectly from here onwards
    @st.cache_resource(show_spinner=False)
    def initialize_service(m_size, dev, comp):
        return TranscriptionService(model_size=m_size, device=dev, compute_type=comp)
        
    try:
        service = initialize_service(model_size, device_choice, compute_choice)
        
        # Setup temporary directories cleanly
        temp_dir = Path("temp_workspace")
        temp_dir.mkdir(exist_ok=True)
