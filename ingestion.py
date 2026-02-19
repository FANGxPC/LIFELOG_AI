import cv2
import threading
import time
import pyaudio
import wave
from faster_whisper import WhisperModel

# --- CONFIGURATION ---
CHUNK_DURATION = 10  # Seconds
FPS_TO_SAVE = 1     # We want 5 frames in 5 seconds (1 frame per second)
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

class LifeLogCollector:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.whisper_model = WhisperModel("small.en", device='cuda', compute_type="int8")
        self.frames = []
        self.is_running = True

    def capture_vision(self):
        """Thread 1: Captures 1 frame every second."""
        
        while self.is_running:
            ret, frame = self.camera.read()
            if ret:
                if len(self.frames) < 10:
                    self.frames.append(frame)
                time.sleep(1) # Wait 1 second to get the next frame

    def capture_audio(self):
        """Thread 2: Records 5 seconds of audio and transcribes."""
        p = pyaudio.PyAudio()
        stream = p.open(format=AUDIO_FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        
        while self.is_running:
            audio_frames = []
            # Record for 5 seconds
            for _ in range(0, int(RATE / CHUNK * CHUNK_DURATION)):
                audio_frames.append(stream.read(CHUNK))
            
            # Save and Transcribe
            with wave.open("temp_audio.wav", "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(AUDIO_FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(audio_frames))
            
            segments, _ = self.whisper_model.transcribe("temp_audio.wav")
            transcript = " ".join([seg.text for seg in segments])
            
            # TRIGGER THE BRAIN
            self.sync_to_brain(transcript)

    def sync_to_brain(self, transcript):
        """Coordinates the hand-off."""
        current_frames = self.frames.copy()
        self.frames = [] # Reset for next 5 seconds
        
        print(f"\n--- 5 SEC SYNC ---")
        print(f"Vision: {len(current_frames)} frames captured.")
        print(f"Audio: '{transcript}'")
        # DATA READY FOR DAY 2 (The Brain)

if __name__ == "__main__":
    collector = LifeLogCollector()

    vision_thread = threading.Thread(target=collector.capture_vision)
    audio_thread = threading.Thread(target=collector.capture_audio)

    vision_thread.start()
    audio_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        collector.is_running = False
        vision_thread.join()
        audio_thread.join()
        collector.camera.release()
        cv2.destroyAllWindows()
        print("Cleaned up and exited.")