import logging
import io
import wave
import numpy as np
from typing import Callable, Awaitable, Optional, List
from heisenberg.interfaces.stt import ABCSTT
from heisenberg.core.config import STTConfig

# Try to import whispercpp, handle missing dependency gracefully
try:
    from whispercpp import Whisper
except ImportError:
    Whisper = None

logger = logging.getLogger(__name__)

class WhisperSTT(ABCSTT):
    """
    STT implementation using whispercpp (GGML models).
    """

    def __init__(self, config: STTConfig):
        self.config = config
        self._whisper: Optional[Whisper] = None
        self._partial_callback: Optional[Callable[[str], Awaitable[None]]] = None
        self._final_callback: Optional[Callable[[str], Awaitable[None]]] = None
        self._buffer = bytearray()
        self._is_running = False

        if Whisper is None:
            logger.error("whispercpp library not found. Please install it with 'pip install whispercpp'.")
        else:
            try:
                self._whisper = Whisper(
                    model_path=self.config.model_path,
                    language=self.config.language,
                    n_threads=self.config.n_threads
                )
                logger.info(f"WhisperSTT initialized with model: {self.config.model_path}")
            except Exception as e:
                logger.error(f"Failed to initialize WhisperSTT: {e}")

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
        logger.info("WhisperSTT session stopped, processing final audio...")
        
        if not self._whisper or not self._buffer:
            return

        try:
            # Convert buffer to numpy array (float32, normalized)
            # whisper.cpp expects 16kHz mono. PyAudioIO provides 16kHz mono int16.
            audio_int16 = np.frombuffer(self._buffer, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            # Since whispercpp's transcribe might not support raw arrays directly in all versions,
            # or it might expect a file path, we check how to call it.
            # Based on user snippet: segments = w.transcribe("audio.wav")
            # If it only takes a file, we might need a temporary WAV file.
            
            # Let's try to use a temporary file to be safe and match the user snippet
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name
                
            try:
                # Write to WAV
                with wave.open(tmp_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2) # 16-bit
                    wf.setframerate(16000)
                    wf.writeframes(self._buffer)
                
                # Transcribe
                logger.debug(f"Calling whispercpp.transcribe on {tmp_path}")
                segments = self._whisper.transcribe(tmp_path)
                logger.debug(f"Transcription complete. Got {len(segments)} segments.")
                
                # Combine segments
                full_text = " ".join([s.text for s in segments]).strip()
                logger.info(f"Full transcription: '{full_text}'")
                
                if self._final_callback:
                    await self._final_callback(full_text)
                    
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
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
