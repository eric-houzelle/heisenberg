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
        import time
        last_log_time = time.time()
        max_score_seen = 0.0
        
        try:
            while self.running:
                frame = await self.audio_source.read_frame()
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue

                # OpenWakeWord expects 16-bit PCM (int16)
                audio_data = np.frombuffer(frame, dtype=np.int16)
                
                # Predict
                # NOTE: The model expects exactly 1280 samples for 80ms of audio
                predictions = self.model.predict(audio_data)
                
                # Track and log max score periodically
                for wakeword, score in predictions.items():
                    max_score_seen = max(max_score_seen, score)
                    
                    if score >= self.config.threshold:
                        logger.info(f"Wakeword detected: {wakeword} (score: {score:.2f})")
                        if self.callback:
                            await self.callback()
                
                if time.time() - last_log_time > 3:
                     if max_score_seen > 0.0001:
                         logger.debug(f"Max wakeword score in last 3s: {max_score_seen:.4f}")
                     max_score_seen = 0.0
                     last_log_time = time.time()
        except Exception as e:
            logger.error(f"Error in OpenWakeWordEngine loop: {e}", exc_info=True)
            self.running = False
