# Naive energy-based VAD for demo only (classroom use is noisy/unreliable).
# Only enable if you run a single-seat quiet setup.

import numpy as np
import pyaudio
import time
from config import AUDIO_SECONDS, AUDIO_THRESHOLD

class SimpleVAD:
    def __init__(self, rate=16000, chunk=1024):
        self.rate = rate
        self.chunk = chunk
        self.pa = pyaudio.PyAudio()
        self.stream = None

    def start(self):
        self.stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=self.rate,
                                   input=True, frames_per_buffer=self.chunk)

    def stop(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()

    def speech_present(self) -> bool:
        """Return True if RMS above threshold during AUDIO_SECONDS."""
        frames = int(self.rate * AUDIO_SECONDS / self.chunk)
        acc = 0.0
        for _ in range(max(1, frames)):
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            acc += float(np.sqrt(np.mean(x*x)))
        rms = acc / max(1, frames)
        return rms > AUDIO_THRESHOLD
