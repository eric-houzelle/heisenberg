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
from heisenberg.stt.whisper import WhisperSTT
from heisenberg.orchestrator.state import State

async def main():
    setup_logging(level="INFO")
    logger = logging.getLogger("main")
    logger.info("Starting Heisenberg...")

    # Load configuration
    config = Config.load()

    # Wire up components
    router = EventRouter()
    fsm = FSM(router=router)
    
    # Audio and Wakeword setup
    audio_source = PyAudioIO(config.audio)
    wakeword_engine = OpenWakeWordEngine(config.wakeword)
    stt_engine = WhisperSTT(config.stt)

    # Register detection callback to FSM
    wakeword_engine.on_detected(
        lambda: fsm.handle_event(Event.WAKEWORD_DETECTED)
    )

    # Register handlers for FSM events
    listening_task = None

    async def stop_listening_after_timeout(seconds: float):
        await asyncio.sleep(seconds)
        if fsm.state == State.LISTENING:
            logger.info(f"Listening timeout reached ({seconds}s). Stopping STT stream.")
            await stt_engine.stop_stream()

    async def on_wakeword():
        nonlocal listening_task
        logger.info("Wakeword detected handler: Starting STT stream")
        await stt_engine.start_stream()
        # Start a timeout task to stop listening
        if listening_task:
            listening_task.cancel()
        listening_task = asyncio.create_task(stop_listening_after_timeout(5.0))

    async def on_transcription_final(text: str):
        logger.info(f"Transcription final: {text}")
        await fsm.handle_event(Event.TRANSCRIPTION_FINAL)
        # Transition back to IDLE after thinking (simulated for now)
        await asyncio.sleep(1)
        await fsm.transition(State.IDLE)

    router.register(Event.WAKEWORD_DETECTED, on_wakeword)
    stt_engine.on_final(on_transcription_final)

    # Handle graceful shutdown
    loop = asyncio.get_running_loop()
    def stop_all():
        logger.info("Stopping...")
        # Create tasks for stopping to avoid blocking
        asyncio.create_task(audio_source.stop())
        asyncio.create_task(wakeword_engine.stop())
        asyncio.create_task(stt_engine.stop_stream())
        # Give it a moment to stop before exiting
        loop.call_later(1, sys.exit, 0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_all)

    # Start loop
    try:
        await audio_source.start()
        await wakeword_engine.start()
        await fsm.start()
        
        logger.info("Main loop started. Listening for wakeword...")
        
        # Central Audio Loop
        
        while True:
            frame = await audio_source.read_frame()
            if frame:
                # Always feed wakeword if IDLE
                if fsm.state == State.IDLE:
                    await wakeword_engine.feed_audio(frame)
                
                # Feed STT if LISTENING
                if fsm.state == State.LISTENING:
                    await stt_engine.feed_audio(frame)
            else:
                await asyncio.sleep(0.01)
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
