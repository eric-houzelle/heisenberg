import asyncio
import logging
import pyaudio
import numpy as np
from typing import Optional
from heisenberg.interfaces.audio import ABCAudioIO
from heisenberg.core.config import AudioConfig

logger = logging.getLogger(__name__)

class PyAudioIO(ABCAudioIO):
    """
    Implementation of ABCAudioIO using PyAudio for microphone capture.
    Supports automatic resampling if the hardware doesn't support the target sample rate.
    """
    def __init__(self, config: AudioConfig):
        self.config = config
        self.pa = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self._loop = asyncio.get_event_loop()
        
        # Target for internal processing (usually 16000)
        self.target_rate = 16000 
        self.actual_rate = self.config.sample_rate
        
        # Async queue for buffered frames
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100) # Buffer max ~8 seconds

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback from PyAudio thread when data is ready."""
        if status:
            logger.warning(f"PyAudio status: {status}")
        
        # Process resampling in the callback thread (fast enough)
        processed_data = self._process_frame(in_data)
        
        # Use call_soon_threadsafe to put in the asyncio queue
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, processed_data)
        except asyncio.QueueFull:
            # If queue is full, we drop oldest frame to maintain "real-time"
            try:
                # We can't actually get() in a callback easily without slowing down
                # But we can try to pop if we were in the loop.
                # Simplest for now: just log overflow
                logger.debug("Audio buffer overflow, dropping frame")
                pass
            except Exception:
                pass
        
        return (None, pyaudio.paContinue)

    def _process_frame(self, data: bytes) -> bytes:
        """Resample frame if needed."""
        if self.actual_rate == self.target_rate:
            return data
            
        # Linear interpolation resampling
        audio_data_int16 = np.frombuffer(data, dtype=np.int16)
        new_len = self.config.chunk_size
        resampled = np.interp(
            np.linspace(0, len(audio_data_int16), new_len, endpoint=False),
            np.arange(len(audio_data_int16)),
            audio_data_int16
        ).astype(np.int16)
        return resampled.tobytes()

    async def start(self) -> None:
        if self.stream:
            return

        idx = self.config.input_device_index if self.config.input_device_index != -1 else None
        
        # Clear queue
        while not self._queue.empty():
            self._queue.get_nowait()

        # Try target rate first
        try:
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.target_rate,
                input=True,
                input_device_index=idx,
                frames_per_buffer=self.config.chunk_size,
                stream_callback=self._audio_callback
            )
            self.actual_rate = self.target_rate
            logger.info(f"PyAudioIO started at target rate: {self.target_rate}Hz (Callback mode)")
        except Exception:
            # Try device default rate
            device_info = self.pa.get_default_input_device_info() if idx is None else self.pa.get_device_info_by_index(idx)
            self.actual_rate = int(device_info['defaultSampleRate'])
            logger.warning(f"Target rate {self.target_rate}Hz unsupported. Falling back to device rate: {self.actual_rate}Hz")
            
            hw_chunk_size = int(self.config.chunk_size * (self.actual_rate / self.target_rate))
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.actual_rate,
                input=True,
                input_device_index=idx,
                frames_per_buffer=hw_chunk_size,
                stream_callback=self._audio_callback
            )
            logger.info(f"PyAudioIO started at fallback rate: {self.actual_rate}Hz (Callback mode)")

        self.stream.start_stream()

    async def stop(self) -> None:
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        logger.info("PyAudioIO stream stopped")

    async def read_frame(self) -> Optional[bytes]:
        """Pop a frame from the internal queue."""
        try:
            return await self._queue.get()
        except Exception:
            return None

    async def play_frame(self, frame: bytes) -> None:
        pass

class AudioCapture:
    """Handles audio input capture."""
    pass
