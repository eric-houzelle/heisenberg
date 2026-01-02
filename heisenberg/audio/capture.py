import asyncio
import logging
import pyaudio
from typing import Optional
from heisenberg.interfaces.audio import ABCAudioIO
from heisenberg.core.config import AudioConfig

logger = logging.getLogger(__name__)

class PyAudioIO(ABCAudioIO):
    """
    Implementation of ABCAudioIO using PyAudio for microphone capture.
    """
    def __init__(self, config: AudioConfig):
        self.config = config
        self.pa = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self._loop = asyncio.get_event_loop()

    async def start(self) -> None:
        if self.stream:
            return
            
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            input_device_index=self.config.input_device_index if self.config.input_device_index != -1 else None,
            frames_per_buffer=self.config.chunk_size
        )
        logger.info("PyAudioIO stream started")

    async def stop(self) -> None:
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        logger.info("PyAudioIO stream stopped")

    async def read_frame(self) -> Optional[bytes]:
        if not self.stream or not self.stream.is_active():
            return None
            
        try:
            # Pyaudio read is blocking, run in thread to avoid blocking event loop
            data = await self._loop.run_in_executor(
                None, 
                self.stream.read, 
                self.config.chunk_size,
                False # exception_on_overflow
            )
            return data
        except Exception as e:
            logger.error(f"Error reading from PyAudio: {e}")
            return None

    async def play_frame(self, frame: bytes) -> None:
        # Implementation for playback can be added if needed
        pass

class AudioCapture:
    """Handles audio input capture."""
    pass
