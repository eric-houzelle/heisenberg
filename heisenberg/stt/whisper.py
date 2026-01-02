import logging
import os
import wave
import numpy as np
from typing import Callable, Awaitable, Optional, List
from heisenberg.interfaces.stt import ABCSTT
from heisenberg.core.config import STTConfig

# Try to import pywhispercpp, handle missing dependency gracefully
try:
    from pywhispercpp.model import Model
except ImportError:
    Model = None

logger = logging.getLogger(__name__)

class WhisperSTT(ABCSTT):
    """
    STT implementation using pywhispercpp (GGML models).
    """

    def __init__(self, config: STTConfig):
        self.config = config
        self._model: Optional[Model] = None
        self._partial_callback: Optional[Callable[[str], Awaitable[None]]] = None
        self._final_callback: Optional[Callable[[str], Awaitable[None]]] = None
        self._buffer = bytearray()
        self._is_running = False

        if Model is None:
            logger.error("pywhispercpp library not found. Please install it with 'pip install pywhispercpp'.")
        else:
            try:
                logger.info(f"Initializing pywhispercpp with model: {self.config.model_path}...")
                self._model = Model(
                    self.config.model_path,
                    n_threads=self.config.n_threads,
                    # Disabling prints as they might interfere with our own logging
                    print_realtime=False,
                    print_progress=False,
                    print_timestamps=False
                )
                logger.info("WhisperSTT (pywhispercpp) successfully initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize WhisperSTT: {e}", exc_info=True)

    async def start_stream(self) -> None:
        """Start the STT streaming session."""
        self._buffer = bytearray()
        self._is_running = True
        logger.info("WhisperSTT session started")

    async def stop_stream(self) -> None:
        """Stop the STT streaming session and trigger final transcription."""
        if not self._is_running:
            return
        
        self._is_running = False
        buffer_len = len(self._buffer)
        logger.info(f"WhisperSTT session stopped. Buffer size: {buffer_len} bytes. Processing final audio...")
        
        if not self._model:
            logger.error("Whisper model not initialized!")
            return
            
        if buffer_len == 0:
            logger.warning("Audio buffer is empty, nothing to transcribe.")
            return

        try:
            # pywhispercpp can take a numpy array directly or a file.
            # Convert buffer to numpy array (float32, normalized)
            # whisper.cpp expects 16kHz mono. PyAudioIO provides 16kHz mono int16.
            audio_int16 = np.frombuffer(self._buffer, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            # Transcription using pywhispercpp
            logger.debug("Calling pywhispercpp.model.transcribe")
            segments = self._model.transcribe(audio_float32, language=self.config.language)
            
            # Combine segments
            full_text = " ".join([s.text for s in segments]).strip()
            logger.info(f"Full transcription: '{full_text}'")
            
            # Optional: Dump audio to WAV for debugging
            if self.config.debug_dump:
                try:
                    import uuid
                    dump_path = f"debug_stt_{uuid.uuid4().hex[:8]}.wav"
                    with wave.open(dump_path, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2) # 16-bit
                        wf.setframerate(16000)
                        wf.writeframes(self._buffer)
                    logger.info(f"Debug audio dumped to: {dump_path}")
                except Exception as e:
                    logger.error(f"Failed to dump debug audio: {e}")

            if self._final_callback:
                await self._final_callback(full_text)
                    
        except Exception as e:
            logger.error(f"Error during transcription: {e}", exc_info=True)
        finally:
            self._buffer = bytearray()

    async def feed_audio(self, frame: bytes) -> None:
        """Feed audio data to the STT engine."""
        if self._is_running:
            self._buffer.extend(frame)

    def on_partial(self, callback: Callable[[str], Awaitable[None]]) -> None:
        """Register callback for partial transcription updates."""
        self._partial_callback = callback

    def on_final(self, callback: Callable[[str], Awaitable[None]]) -> None:
        """Register callback for final transcription results."""
        self._final_callback = callback
