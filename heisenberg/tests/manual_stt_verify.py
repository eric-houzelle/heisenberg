import asyncio
import argparse
import logging
import os
from heisenberg.stt.whisper import WhisperSTT
from heisenberg.core.config import STTConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    parser = argparse.ArgumentParser(description="Manual verification for Whisper STT")
    parser.add_argument("--model", type=str, required=True, help="Path to the GGML model file")
    parser.add_argument("--audio", type=str, required=True, help="Path to a 16kHz WAV file")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        return

    if not os.path.exists(args.audio):
        logger.error(f"Audio file not found: {args.audio}")
        return

    config = STTConfig(model_path=args.model)
    stt = WhisperSTT(config)

    async def on_final(text):
        print(f"\n[FINAL TRANSCRIPTION]: {text}\n")

    stt.on_final(on_final)

    print(f"Opening audio file: {args.audio}")
    import wave
    with wave.open(args.audio, 'rb') as wf:
        if wf.getframerate() != 16000:
            logger.warning(f"Audio file framerate is {wf.getframerate()}Hz, expected 16000Hz. This might fail.")
        
        await stt.start_stream()
        
        data = wf.readframes(1024)
        while data:
            await stt.feed_audio(data)
            data = wf.readframes(1024)
            # Small sleep to simulate streaming if needed, though not strictly necessary for this test
            
        await stt.stop_stream()

if __name__ == "__main__":
    asyncio.run(main())
