import asyncio
import logging
import signal
import sys
from heisenberg.core.config import Config
from heisenberg.audio.capture import PyAudioIO
from heisenberg.wakeword.engine import OpenWakeWordEngine

async def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("rpi_test")
    logger.info("Starting Heisenberg Wake Word test on RPi...")

    # Load configuration
    config = Config.load()
    
    # Initialize components
    audio_source = PyAudioIO(config.audio)
    wakeword_engine = OpenWakeWordEngine(config.wakeword, audio_source)

    # Detection callback
    async def on_detected():
        print("\n" + "="*40)
        print(">>> WAKEWORD DETECTED: HEY JARVIS! <<<")
        print("="*40 + "\n")

    wakeword_engine.on_detected(on_detected)

    # Handle graceful shutdown
    loop = asyncio.get_running_loop()
    def stop_all():
        logger.info("Stopping...")
        asyncio.create_task(audio_source.stop())
        asyncio.create_task(wakeword_engine.stop())
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_all)

    # Start audio and engine
    try:
        await audio_source.start()
        await wakeword_engine.start()
        
        logger.info("System is listening... (Press Ctrl+C to stop)")
        while True:
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"Error during runtime: {e}")
    finally:
        await audio_source.stop()
        await wakeword_engine.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
