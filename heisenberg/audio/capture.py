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
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._running = False
        self._capture_thread: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self.stream:
            return

        idx = self.config.input_device_index if self.config.input_device_index != -1 else None
        
        # Try target rate first
        try:
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.target_rate,
                input=True,
                input_device_index=idx,
                frames_per_buffer=self.config.chunk_size
            )
            self.actual_rate = self.target_rate
            logger.info(f"PyAudioIO started at target rate: {self.target_rate}Hz")
        except Exception:
            # Try device default rate
            device_info = self.pa.get_default_input_device_info() if idx is None else self.pa.get_device_info_by_index(idx)
            self.actual_rate = int(device_info['defaultSampleRate'])
            logger.warning(f"Target rate {self.target_rate}Hz unsupported. Falling back to device rate: {self.actual_rate}Hz")
            
            # Use a larger buffer for the hardware stream when resampling
            hw_chunk_size = int(self.config.chunk_size * (self.actual_rate / self.target_rate))
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.actual_rate,
                input=True,
                input_device_index=idx,
                frames_per_buffer=hw_chunk_size * 2 # Double buffer to avoid overflows
            )
            logger.info(f"PyAudioIO started at fallback rate: {self.actual_rate}Hz")

        self._running = True
        self._capture_thread = asyncio.create_task(self._capture_loop())

    async def _capture_loop(self):
        """Continuously read from the stream and push to the queue."""
        logger.debug("Audio capture loop started")
        read_size = self.config.chunk_size
        if self.actual_rate != self.target_rate:
            read_size = int(self.config.chunk_size * (self.actual_rate / self.target_rate))

        while self._running and self.stream:
            try:
                # Read blockingly in a thread pool to avoid blocking the event loop
                data = await self._loop.run_in_executor(
                    None, 
                    self.stream.read, 
                    read_size,
                    False # exception_on_overflow=False
                )
                
                if data:
                    processed_data = self._process_frame(data)
                    await self._queue.put(processed_data)
                    
            except Exception as e:
                if self._running:
                    logger.error(f"Error in audio capture loop: {e}")
                await asyncio.sleep(0.1)
        logger.debug("Audio capture loop stopped")

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

    async def stop(self) -> None:
        self._running = False
        if self._capture_thread:
            self._capture_thread.cancel()
            try:
                await self._capture_thread
            except asyncio.CancelledError:
                pass
            self._capture_thread = None

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
            # Using wait_for to avoid blocking forever if something breaks
            return await asyncio.wait_for(self._queue.get(), timeout=1.0)
        except (asyncio.TimeoutError, Exception):
            return None

    async def play_frame(self, frame: bytes) -> None:
        pass

class AudioCapture:
    """Handles audio input capture."""
    pass
