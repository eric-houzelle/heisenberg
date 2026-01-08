import logging
import torch
import numpy as np
from typing import Optional
from heisenberg.core.config import VADConfig

logger = logging.getLogger(__name__)

class SileroVADEngine:
    """
    Voice Activity Detection using Silero VAD.
    Optimized for 16kHz audio.
    """
    def __init__(self, config: VADConfig):
        self.config = config
        self.model = None
        self._utils = None
        
        # Load model using torch hub
        try:
            logger.info("Loading Silero VAD model...")
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            self.model = model
            self._utils = utils
            logger.info("Silero VAD model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")

        self._reset()

    def _reset(self):
        self._is_speaking = False
        self._silence_frames = 0
        self._speech_frames = 0
        self._frames_per_ms = 16 # 16000 / 1000
        self._buffer = b""
        
    def is_speech(self, frame: bytes) -> bool:
        """
        Detects if the given 16kHz frame contains speech.
        Returns True if the system is currently in a "Speaking" state (including lead-out silence).
        """
        if self.model is None:
            return True # Fail-safe: assume speech if model is missing
            
        try:
            self._buffer += frame
            
            # Silero VAD requires chunks of 512 samples (1024 bytes) for 16kHz
            while len(self._buffer) >= 1024:
                chunk_bytes = self._buffer[:1024]
                self._buffer = self._buffer[1024:]
                
                audio_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                
                # Silero expects torch tensor
                input_tensor = torch.from_numpy(audio_float32)
                
                # Get speech probability
                # The model returns a probability tensor
                speech_prob = self.model(input_tensor, 16000).item()
                
                # Logic for start/end detection with hysteresis
                if speech_prob > self.config.threshold:
                    self._speech_frames += 1
                    self._silence_frames = 0
                    if not self._is_speaking and self._speech_frames >= 2: # Small debouncing
                        self._is_speaking = True
                        logger.debug("VAD: Speech started")
                else:
                    self._silence_frames += 1
                    self._speech_frames = 0
                    
                    # Check for silence timeout
                    # Each chunk is 512 samples -> 32ms
                    silence_ms = (self._silence_frames * 512) / self._frames_per_ms
                    if self._is_speaking and silence_ms > self.config.min_silence_duration_ms:
                        self._is_speaking = False
                        logger.debug(f"VAD: Speech ended (silence: {silence_ms:.0f}ms)")
            
            return self._is_speaking
            
        except Exception as e:
            logger.error(f"Error in VAD processing: {e}", exc_info=True)
            return True # Fail-safe

    def reset(self):
        """Reset internal VAD state."""
        self._reset()
