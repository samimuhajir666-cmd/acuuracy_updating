import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel

class TranscriptionService:
    """Production-ready transcription service using faster-whisper with Live Recording support."""
    
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        """Initialize the transcription service."""
        print(f"Loading model: {model_size} on {device} ({compute_type})")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
        print("Model loaded successfully!\n")

    def record_audio(self, output_path="recorded_audio.wav", duration=None, sample_rate=16000):
        """
        Record audio from microphone.
        
        Args:
            output_path: Path to save the recorded WAV file
            duration: Recording time in seconds (if None, press Enter to stop)
            sample_rate: Audio sampling rate (default 16kHz for Whisper)
        """
        print("🎙️ Preparing Microphone...")
        
        if duration:
            print(f"🔴 Recording started for {duration} seconds...")
            recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
            sd.wait()
            print("⏹️ Recording finished!")
        else:
            # Manual stop recording mode
            audio_data = []
            def callback(indata, frames, time, status):
                if status:
                    print(f"Status error: {status}")
                audio_data.append(indata.copy())

            print("🔴 Recording started! Press ENTER to stop recording...")
            stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16', callback=callback)
            
            with stream:
                input() # Waits for user to press Enter
                
            print("⏹️ Recording stopped!")
            recording = np.concatenate(audio_data, axis=0)

        # Save to file
        output_file = Path(output_path)
        wav.write(output_file, sample_rate, recording)
        print(f"📁 Audio saved to: {output_file.name}")
        return output_file
    
    def transcribe_file(self, audio_path, output_format="txt", **kwargs):
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to audio file
            output_format: Output format (txt, json, srt, vtt)
            **kwargs: Additional transcription parameters
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"Transcribing: {audio_path.name}")
        
        # Transcribe
        segments, info = self.model.transcribe(
            str(audio_path),
            word_timestamps=True,
            **kwargs
        )
        
        # Collect results
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
        
        # Save based on format
        output_path = audio_path.parent / f"{audio_path.stem}_transcript"
        
        if output_format == "txt":
            self._save_txt(result, output_path.with_suffix(".txt"))
        elif output_format == "json":
            self._save_json(result, output_path.with_suffix(".json"))
        elif output_format == "srt":
            self._save_srt(result, output_path.with_suffix(".srt"))
        elif output_format == "vtt":
            self._save_vtt(result, output_path.with_suffix(".vtt"))
        
        print(f"✓ Transcription saved: {output_path}.{output_format}")
        return result
    
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


# Usage Example
if __name__ == "__main__":
    # 1. Initialize service
    service = TranscriptionService(
        model_size="base",
        device="cpu",      # "cuda" if GPU is available
        compute_type="int8"
    )
    
    # 2. Record live audio from microphone
    # Note: duration=None rakha hai taaki jab tak aap ENTER na dabayein tab tak recording ho.
    # Agar fix seconds chahiye toh `duration=10` pass kardein.
    recorded_file = service.record_audio(
        output_path="my_recording.wav",
        duration=None 
    )
    
    # 3. Transcribe the recorded file
    result = service.transcribe_file(
        recorded_file,
        output_format="txt",
        beam_size=5
    )
    
    # 4. Display result
    print("\n--- TRANSCRIPTION RESULT ---")
    print(f"Detected Language: {result['language']}")
    print(f"Duration: {result['duration']:.2f} seconds")
    print(f"Full Text: {result['text']}")
