import pytest
import asyncio
from unittest.mock import MagicMock, patch
from heisenberg.stt.whisper import WhisperSTT
from heisenberg.core.config import STTConfig

@pytest.fixture
def stt_config():
    return STTConfig(model_path="fake_model.bin")

@pytest.mark.asyncio
async def test_whisper_stt_flow(stt_config):
    # Mock whispercpp
    with patch("heisenberg.stt.whisper.Whisper") as MockWhisper:
        mock_instance = MockWhisper.return_value
        # Mock segments
        mock_segment = MagicMock()
        mock_segment.text = "Hello world"
        mock_instance.transcribe.return_value = [mock_segment]
        
        stt = WhisperSTT(stt_config)
        
        # Callbacks
        final_text = ""
        async def on_final(text):
            nonlocal final_text
            final_text = text
            
        stt.on_final(on_final)
        
        # Simulate stream
        await stt.start_stream()
        # Feed 160 samples (320 bytes for 16-bit)
        await stt.feed_audio(bytes([0] * 320))
        await stt.stop_stream()
        
        # Verify
        assert final_text == "Hello world"
        assert MockWhisper.called
        assert mock_instance.transcribe.called
