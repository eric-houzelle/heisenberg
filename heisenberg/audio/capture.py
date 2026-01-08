import asyncio
import logging
import pyaudio
import numpy as np
from typing import Optional
from heisenberg.interfaces.audio import ABCAudioIO
from heisenberg.core.config import AudioConfig

# Try to import pyrnnoise for noise suppression
try:
    from pyrnnoise import RNNoise
except ImportError:
    RNNoise = None

logger = logging.getLogger(__name__)

class PyAudioIO(ABCAudioIO):
    """
    Implementation of ABCAudioIO using PyAudio for microphone capture.
    Optimized for RNNoise (48kHz capture) and STT/Wakeword (16kHz processing).
    """
    def __init__(self, config: AudioConfig):
        self.config = config
        self.pa = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self._loop = asyncio.get_event_loop()
        
        # Internal target for Heisenberg (fixed at 16000)
        self.process_rate = 16000 
        
        # Hardware target: RNNoise prefers 48000. 
        # If RNNoise is not available, we can drop to 16000 to save CPU.
        self.hardware_rate = 48000 if RNNoise else self.process_rate
        self.actual_rate = self.hardware_rate # Will be updated on start()
        
        # RNNoise setup
        self._denoiser: Optional[RNNoise] = None
        if RNNoise:
            try:
                self._denoiser = RNNoise()
                logger.info("RNNoise denoiser initialized (48kHz)")
            except Exception as e:
                logger.error(f"Failed to initialize RNNoise: {e}")
        
        # Async queue for buffered frames (ready for consumption at 16kHz)
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback from PyAudio thread when data is ready."""
        if status:
            if status & pyaudio.paInputOverflow:
                logger.debug(f"PyAudio input overflow (status: {status})")
            else:
                logger.warning(f"PyAudio status: {status}")
        
        # Process audio in the callback thread
        processed_data = self._process_frame_pipeline(in_data, frame_count)
        
        # Push to the asyncio queue
        if processed_data:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, processed_data)
        
        return (None, pyaudio.paContinue)

    def _process_frame_pipeline(self, data: bytes, frame_count: int) -> bytes:
        """
        The Audio Processing Pipeline:
        1. Decode to int16
        2. Resample to 48kHz (required for RNNoise)
        3. Clean with RNNoise
        4. Resample to 16kHz (required for Wakeword/STT)
        """
        audio_int16 = np.frombuffer(data, dtype=np.int16)
        if len(audio_int16) == 0:
            return b""

        # --- Stage 1: Ensure 48kHz for RNNoise ---
        if self.actual_rate != 48000:
            # Upsample/Downsample to 48k
            target_len = int(len(audio_int16) * (48000 / self.actual_rate))
            if target_len > 0:
                audio_48k = np.interp(
                    np.linspace(0, len(audio_int16), target_len, endpoint=False),
                    np.arange(len(audio_int16)),
                    audio_int16
                ).astype(np.int16)
            else:
                return b""
        else:
            audio_48k = audio_int16

        # --- Stage 2: RNNoise Denoising ---
        # RNNoise operates on 10ms (480 samples @ 48kHz) chunks.
        if self._denoiser:
            cleaned_chunks = []
            # logical chunks of 480
            for i in range(0, len(audio_48k), 480):
                chunk = audio_48k[i:i+480]
                if len(chunk) == 480:
                    cleaned_chunks.append(self._denoiser.denoise_frame(chunk.tobytes()))
                else:
                    # Keep partial chunk without denoising (too small)
                    cleaned_chunks.append(chunk.tobytes())
            audio_48k = np.frombuffer(b"".join(cleaned_chunks), dtype=np.int16)

        # --- Stage 3: Resample to 16kHz for System ---
        # We always want 16kHz output
        target_len_16k = int(len(audio_48k) * (16000 / 48000))
        if target_len_16k > 0:
            audio_16k = np.interp(
                np.linspace(0, len(audio_48k), target_len_16k, endpoint=False),
                np.arange(len(audio_48k)),
                audio_48k
            ).astype(np.int16)
        else:
            return b""

        return audio_16k.tobytes()

    async def start(self) -> None:
        if self.stream:
            return

        idx = self.config.input_device_index if self.config.input_device_index != -1 else None
        
        # Clear queue
        while not self._queue.empty():
            self._queue.get_nowait()

        # Try hardware rate (48k or 16k)
        try:
            # OpenWakeWord requires at least 80ms (1280 samples @ 16kHz) for optimal performance.
            chunk_size = (self.hardware_rate * 8) // 100 
            
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.hardware_rate,
                input=True,
                input_device_index=idx,
                frames_per_buffer=chunk_size,
                stream_callback=self._audio_callback
            )
            self.actual_rate = self.hardware_rate
            logger.info(f"PyAudioIO started at {self.actual_rate}Hz (RNNoise: {bool(self._denoiser)})")
        except Exception as e:
            logger.warning(f"Failed to start at {self.hardware_rate}Hz: {e}. Falling back to device default.")
            # Final fallback to device default
            device_info = self.pa.get_default_input_device_info() if idx is None else self.pa.get_device_info_by_index(idx)
            self.actual_rate = int(device_info['defaultSampleRate'])
            
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.actual_rate,
                input=True,
                input_device_index=idx,
                frames_per_buffer=int(self.actual_rate * 8 / 100), # 80ms
                stream_callback=self._audio_callback
            )
            logger.info(f"PyAudioIO started at fallback rate: {self.actual_rate}Hz")

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
        """Pop a 16kHz frame from the internal queue."""
        try:
            return await self._queue.get()
        except Exception:
            return None

    async def play_frame(self, frame: bytes) -> None:
        pass
