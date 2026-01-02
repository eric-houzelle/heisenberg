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
            
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.actual_rate,
                input=True,
                input_device_index=idx,
                frames_per_buffer=int(self.config.chunk_size * (self.actual_rate / self.target_rate))
            )
            logger.info(f"PyAudioIO started at fallback rate: {self.actual_rate}Hz")

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
        if not self.stream:
            return None
            
        try:
            # Adjust chunk size based on actual rate
            read_size = self.config.chunk_size
            if self.actual_rate != self.target_rate:
                read_size = int(self.config.chunk_size * (self.actual_rate / self.target_rate))

            data = await self._loop.run_in_executor(
                None, 
                self.stream.read, 
                read_size,
                False 
            )

            # Debug: Log RMS level to check if audio is active
            audio_data_int16 = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(np.square(audio_data_int16.astype(np.float32))))
            # if rms > 10: # Only log if there's some sound to avoid spam
            #      logger.debug(f"Audio RMS Level: {rms:.2f}")

            if self.actual_rate != self.target_rate:
                # Resample using numpy
                new_len = self.config.chunk_size
                resampled = np.interp(
                    np.linspace(0, len(audio_data_int16), new_len, endpoint=False),
                    np.arange(len(audio_data_int16)),
                    audio_data_int16
                ).astype(np.int16)
                return resampled.tobytes()
            
            return data
        except Exception as e:
            logger.error(f"Error reading from PyAudio: {e}")
            return None

    async def play_frame(self, frame: bytes) -> None:
        pass

class AudioCapture:
    """Handles audio input capture."""
    pass
