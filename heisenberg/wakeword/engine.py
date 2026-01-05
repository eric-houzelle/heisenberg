import asyncio
import logging
import numpy as np
import openwakeword
from typing import Callable, Awaitable, Optional
from heisenberg.interfaces.wakeword import ABCWakeword
from heisenberg.interfaces.audio import ABCAudioIO
from heisenberg.core.config import WakewordConfig

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

    def on_detected(self, callback: Callable[[], Awaitable[None]]) -> None:
        self.callback = callback

    async def start(self) -> None:
        self.running = True
        logger.info("OpenWakeWordEngine started")

    async def stop(self) -> None:
        self.running = False
        logger.info("OpenWakeWordEngine stopped")

    async def feed_audio(self, frame: bytes) -> None:
        """Feed audio data to the wakeword engine."""
        if not self.running:
            return

        try:
            # OpenWakeWord expects 16-bit PCM (int16)
            audio_data = np.frombuffer(frame, dtype=np.int16)
            
            # Predict
            predictions = self.model.predict(audio_data)
            
            # Check detections
            for wakeword, score in predictions.items():
                logger.debug(f"Wakeword score: {score:.2f}") 
                if score >= self.config.threshold:
                    logger.info(f"Wakeword detected: {wakeword} (score: {score:.2f})")
                    if self.callback:
                        await self.callback()
        except Exception as e:
            logger.error(f"Error in OpenWakeWordEngine processing: {e}", exc_info=True)
