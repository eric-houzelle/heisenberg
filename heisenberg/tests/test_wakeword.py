import pytest
import asyncio
import numpy as np
from unittest.mock import MagicMock, AsyncMock
from heisenberg.wakeword.engine import OpenWakeWordEngine
from heisenberg.core.config import WakewordConfig
from heisenberg.interfaces.audio import ABCAudioIO

@pytest.mark.asyncio
async def test_wakeword_engine_initialization():
    config = WakewordConfig(models=["hey_jarvis"])
    audio_source = MagicMock(spec=ABCAudioIO)
    engine = OpenWakeWordEngine(config, audio_source)
    assert engine.model is not None
    assert engine.config.models == ["hey_jarvis"]

@pytest.mark.asyncio
async def test_wakeword_engine_start_stop():
    config = WakewordConfig(models=["hey_jarvis"])
    audio_source = MagicMock(spec=ABCAudioIO)
    audio_source.read_frame = AsyncMock(return_value=None)
    
    engine = OpenWakeWordEngine(config, audio_source)
    await engine.start()
    assert engine.running is True
    await asyncio.sleep(0.1)
    await engine.stop()
    assert engine.running is False

@pytest.mark.asyncio
async def test_wakeword_detection_mock():
    # This test is harder to do without real audio, 
    # but we can at least check that predict is called.
    config = WakewordConfig(models=["hey_jarvis"])
    audio_source = MagicMock(spec=ABCAudioIO)
    
    # Simulate one frame of silence (1024 samples = 2048 bytes for int16)
    silence_frame = bytes(2048)
    audio_source.read_frame = AsyncMock(side_effect=[silence_frame, None])
    
    engine = OpenWakeWordEngine(config, audio_source)
    engine.model.predict = MagicMock(return_value={"hey_jarvis": 0.1})
    
    await engine.start()
    await asyncio.sleep(0.2) # Let it run a bit
    await engine.stop()
    
    assert engine.model.predict.called
