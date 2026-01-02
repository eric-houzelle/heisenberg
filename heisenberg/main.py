import asyncio
import logging
import signal
import sys
from heisenberg.core.logging import setup_logging
from heisenberg.core.config import Config
from heisenberg.orchestrator.fsm import FSM
from heisenberg.orchestrator.router import EventRouter
from heisenberg.orchestrator.events import Event
from heisenberg.audio.capture import PyAudioIO
from heisenberg.wakeword.engine import OpenWakeWordEngine

async def main():
    setup_logging()
    logger = logging.getLogger("main")
    logger.info("Starting Heisenberg...")

    # Load configuration
    config = Config.load()

    # Wire up components
    router = EventRouter()
    fsm = FSM(router=router)
    
    # Audio and Wakeword setup
    audio_source = PyAudioIO(config.audio)
    wakeword_engine = OpenWakeWordEngine(config.wakeword, audio_source)

    # Register detection callback to FSM
    wakeword_engine.on_detected(
        lambda: fsm.handle_event(Event.WAKEWORD_DETECTED)
    )

    # Handle graceful shutdown
    loop = asyncio.get_running_loop()
    def stop_all():
        logger.info("Stopping...")
        # Create tasks for stopping to avoid blocking
        asyncio.create_task(audio_source.stop())
        asyncio.create_task(wakeword_engine.stop())
        # Give it a moment to stop before exiting
        loop.call_later(1, sys.exit, 0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_all)

    # Start loop
    try:
        await audio_source.start()
        await wakeword_engine.start()
        await fsm.start()
        
        # Keep alive
        while True:
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
    finally:
        await audio_source.stop()
        await wakeword_engine.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
