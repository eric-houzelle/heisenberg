import asyncio
import logging
import numpy as np
import openwakeword
from typing import Callable, Awaitable, Optional
from heisenberg.interfaces.wakeword import ABCWakeword
from heisenberg.interfaces.audio import ABCAudioIO
from heisenberg.core.config import WakewordConfig
import wave

logger = logging.getLogger(__name__)

class OpenWakeWordEngine(ABCWakeword):
    def __init__(self, config: WakewordConfig):
        self.config = config
        self.callback: Optional[Callable[[], Awaitable[None]]] = None
        self.running = False
        
        # Resolve model paths
        pretrained_models = openwakeword.get_pretrained_model_paths()
        resolved_model_paths = []
        for model_name in self.config.models:
            # Check if it's already a path
            if model_name.endswith(".onnx") or model_name.endswith(".tflite"):
                resolved_model_paths.append(model_name)
            else:
                # Try to find in pretrained models
                found = False
                for p in pretrained_models:
                    if model_name in p:
                        resolved_model_paths.append(p)
                        found = True
                        break
                if not found:
                    logger.warning(f"Model {model_name} not found in pretrained models and is not a direct path.")
                    resolved_model_paths.append(model_name) # Fallback

        # Initialize openWakeWord model
        self.model = openwakeword.Model(
            wakeword_model_paths=resolved_model_paths,
        )
        logger.info(f"OpenWakeWordEngine initialized with models: {resolved_model_paths}")

        # Debug: Record audio to file
        self.debug_wav_path = "/tmp/wakeword_debug.wav"
        try:
            self.debug_wav = wave.open(self.debug_wav_path, "wb")
            self.debug_wav.setnchannels(1)
            self.debug_wav.setsampwidth(2) # 16-bit
            self.debug_wav.setframerate(16000)
            logger.info(f"DEBUG: Recording wakeword audio to {self.debug_wav_path}")
        except Exception as e:
            logger.error(f"Failed to open debug WAV: {e}")
            self.debug_wav = None

    def on_detected(self, callback: Callable[[], Awaitable[None]]) -> None:
        self.callback = callback

    async def start(self) -> None:
        self.running = True
        logger.info("OpenWakeWordEngine started")

    async def stop(self) -> None:
        self.running = False
        if self.debug_wav:
            try:
                self.debug_wav.close()
            except:
                pass
            self.debug_wav = None
        logger.info("OpenWakeWordEngine stopped")

    async def feed_audio(self, frame: bytes) -> None:
        """Feed audio data to the wakeword engine."""
        if not self.running:
            return

        try:
            # OpenWakeWord expects 16-bit PCM (int16)
            audio_data = np.frombuffer(frame, dtype=np.int16)
            
            # Debug: Write to file
            if self.debug_wav:
                self.debug_wav.writeframes(frame)

            # Predict
            predictions = self.model.predict(audio_data)
            
            # Check detections
            for wakeword, score in predictions.items():
                # DEBUG: Always print score
                print(f"DEBUG: Wakeword '{wakeword}' score: {score:.4f}")
                logger.debug(f"Wakeword score: {score:.2f}") 
                if score >= self.config.threshold:
                    logger.info(f"Wakeword detected: {wakeword} (score: {score:.2f})")
                    if self.callback:
                        await self.callback()
        except Exception as e:
            logger.error(f"Error in OpenWakeWordEngine processing: {e}", exc_info=True)
