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
    def __init__(self, config: WakewordConfig, audio_source: ABCAudioIO):
        self.config = config
        self.audio_source = audio_source
        self.callback: Optional[Callable[[], Awaitable[None]]] = None
        self.running = False
        self._task: Optional[asyncio.Task] = None
        
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
        if self.running:
            return
        self.running = True
        self._task = asyncio.create_task(self._run())
        logger.info("OpenWakeWordEngine started")

    async def stop(self) -> None:
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("OpenWakeWordEngine stopped")

    async def _run(self) -> None:
        try:
            while self.running:
                frame = await self.audio_source.read_frame()
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue

                # Convert bytes to numpy array (assuming 16-bit PCM)
                audio_data = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Predict
                predictions = self.model.predict(audio_data)

                logger.debug(f"Predictions: {predictions}")
                # Debug: log all scores if there's any activity
                if any(s > 0.05 for s in predictions.values()):
                    logger.debug(f"Predictions: {predictions}")

                # Check detections
                for wakeword, score in predictions.items():
                    if score >= self.config.threshold:
                        logger.info(f"Wakeword detected: {wakeword} (score: {score:.2f})")
                        if self.callback:
                            await self.callback()
                            # Optional: sleep or reset to avoid multiple detections for same event
                            # self.model.reset() 
        except Exception as e:
            logger.error(f"Error in OpenWakeWordEngine loop: {e}", exc_info=True)
            self.running = False
