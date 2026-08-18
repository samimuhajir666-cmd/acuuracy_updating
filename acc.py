import os
import json
from pathlib import Path
from datetime import datetime
import streamlit as st
from faster_whisper import WhisperModel

# ==============================================================================
# 🎙️ TRANSCRIPTION SERVICE CLASS
# ==============================================================================
class TranscriptionService:
    """Production-ready transcription service using faster-whisper."""
    
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        """Initialize the transcription service."""
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
    
    def transcribe_file(self, audio_path, output_format="txt", **kwargs):
        """Transcribe an audio file and save output in desired format."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Transcribe with faster-whisper
        segments, info = self.model.transcribe(
            str(audio_path),
            word_timestamps=True,
            **kwargs
        )
        
        # Structure result dictionary
        result = {
            "file": str(audio_path),
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
        
        # Output generation paths
        output_path = audio_path.parent / f"{audio_path.stem}_transcript"
        
        if output_format == "txt":
            formatted_data = result["text"]
            mime_type = "text/plain"
            ext = ".txt"
        elif output_format == "json":
            formatted_data = json.dumps(result, indent=2, ensure_ascii=False)
            mime_type = "application/json"
            ext = ".json"
        elif output_format == "srt":
            formatted_data = self._generate_srt(result)
            mime_type = "text/plain"
            ext = ".srt"
        elif output_format == "vtt":
            formatted_data = self._generate_vtt(result)
            mime_type = "text/vtt"
            ext = ".vtt"

        return result, formatted_data, mime_type, f"{output_path.stem}{ext}"

    def _generate_srt(self, result):
        output = []
        for i, seg in enumerate(result["segments"], start=1):
            start = self._format_srt_time(seg["start"])
            end = self._format_srt_time(seg["end"])
            output.append(f"{i}\n{start} --> {end}\n{seg['text']}\n")
        return "\n".join(output)

    def _generate_vtt(self, result):
        output = ["WEBVTT\n"]
        for seg in result["segments"]:
            start = self._format_vtt_time(seg["start"])
            end = self._format_vtt_time(seg["end"])
            output.append(f"{start} --> {end}\n{seg['text']}\n")
        return "\n".join(output)

    def _format_srt_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _format_vtt_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

# ==============================================================================
# 🖥️ STREAMLIT UI & LIVE RECORDING WORKFLOW
# ==============================================================================
st.set_page_config(page_title="AI Voice Transcriber", page_icon="🎙️", layout="centered")

st.title("🎙️ Live Voice Recording & Transcription")
st.write("Apni aawaz record karein aur Whisper AI se instantly text mein convert karein.")

# Load Model with Cache
@st.cache_resource(show_spinner=False)
def get_service():
    return TranscriptionService(model_size="base", device="cpu", compute_type="int8")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
output_format = st.sidebar.selectbox("Select Export Format", ["txt", "json", "srt", "vtt"])

# Streamlit Browser-based Live Microphone Input
audio_recording = st.audio_input("Microphone se audio record karein")

if audio_recording is not None:
    st.audio(audio_recording)
    
    if st.button("🚀 Transcribe Audio", type="primary"):
        temp_dir = Path("temp_workspace")
        temp_dir.mkdir(exist_ok=True)
        
        temp_audio_path = temp_dir / "live_recording.wav"
        
        # Save audio byte buffer
        with open(temp_audio_path, "wb") as f:
            f.write(audio_recording.getbuffer())
            
        try:
            with st.spinner("⚡ AI model audio transcribe kar raha hai..."):
                service = get_service()
                result, formatted_data, mime_type, file_name = service.transcribe_file(
                    temp_audio_path,
                    output_format=output_format,
                    beam_size=5
                )
            
            st.success("✅ Transcription Complete!")
            
            # Display Details
            col1, col2 = st.columns(2)
            col1.metric("Detected Language", result["language"].upper())
            col2.metric("Duration", f"{result['duration']:.2f} sec")
            
            # Transcribed Content View
            st.subheader("📝 Transcribed Text:")
            st.text_area("Output", value=result["text"], height=200)
            
            # Download Button
            st.download_button(
                label=f"📥 Download ({output_format.upper()})",
                data=formatted_data,
                file_name=file_name,
                mime=mime_type
            )
            
            # Cleanup temp file
            if temp_audio_path.exists():
                os.remove(temp_audio_path)

        except Exception as e:
            st.error(f"❌ Error occurred: {str(e)}")
