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
from heisenberg.llm.stream import LlamaCppLLM
from heisenberg.llm.prompts import PromptBuilder

async def main():
    setup_logging(level="INFO")
    logger = logging.getLogger("main")
    logger.info("Starting Heisenberg...")

    # Load configuration
    config = Config.load()

    # Wire up components
    router = EventRouter()
    fsm = FSM(router=router)
    
    # Audio, VAD and Engines setup
    audio_source = PyAudioIO(config.audio)
    wakeword_engine = OpenWakeWordEngine(config.wakeword)
    stt_engine = WhisperSTT(config.stt)
    vad_engine = None
    if config.vad.enabled:
        from heisenberg.audio.vad import SileroVADEngine
        vad_engine = SileroVADEngine(config.vad)
    
    # LLM setup
    prompt_builder = PromptBuilder(
        system_prompt=config.llm.system_prompt,
        format_style="plain"  # Adjust based on your LFM2 model format
    )
    llm_engine = LlamaCppLLM(config.llm, prompt_builder)
    
    # State variables
    was_speaking = False
    listening_task = None
    current_user_query = None
    llm_response = ""

    # Register detection callback to FSM
    async def _on_wakeword_detected():
        await fsm.handle_event(Event.WAKEWORD_DETECTED)
        
    wakeword_engine.on_detected(_on_wakeword_detected)

    async def stop_listening_after_timeout(seconds: float):
        try:
            await asyncio.sleep(seconds)
            if fsm.state == State.LISTENING:
                logger.info(f"Fail-safe timeout reached ({seconds}s). Force stopping STT.")
                await stt_engine.stop_stream()
        except asyncio.CancelledError:
            pass

    async def on_wakeword():
        nonlocal was_speaking, listening_task, current_user_query, llm_response
        logger.info("Wakeword detected handler: Starting STT stream")
        was_speaking = False
        current_user_query = None
        llm_response = ""
        await stt_engine.start_stream()
        if vad_engine:
            vad_engine.reset()
        
        # Fail-safe timeout (e.g., 10 seconds)
        if listening_task:
            listening_task.cancel()
        listening_task = asyncio.create_task(stop_listening_after_timeout(10.0))

    async def on_transcription_final(text: str):
        nonlocal current_user_query, llm_response
        current_user_query = text
        logger.info(f"Transcription final: {text}")
        
        try:
            # Transition to THINKING state
            await fsm.handle_event(Event.TRANSCRIPTION_FINAL)

            # Stop audio capture to prevent queue overflow during blocking LLM generation
            # This matches the user's suggestion to "stop listening while thinking"
            await audio_source.stop()
            
            # Get conversation history from session manager
            history = fsm.session_manager.get_conversation_history(
                max_turns=config.llm.max_history_turns
            )
            
            # Start LLM generation with streaming
            logger.info("Starting LLM generation...")
            llm_response = ""
            first_token = True
            
            async for token in llm_engine.generate(text, conversation_history=history):
                llm_response += token
                
                # Emit LLM_TOKEN event for first token (for latency tracking)
                if first_token:
                    logger.info("First LLM token received")
                    await router.dispatch(Event.LLM_TOKEN, token)
                    first_token = False
                
                # Here you could feed tokens to TTS for streaming playback
                # TODO: Implement TTS streaming
            
            logger.info(f"LLM generation complete: {llm_response[:100]}...")
            
            # Emit LLM_COMPLETE event
            await router.dispatch(Event.LLM_COMPLETE, llm_response)
            
            # Store conversation turn in session
            fsm.session_manager.add_conversation_turn(current_user_query, llm_response)
            
            # TODO: Once TTS is implemented, wait for TTS_COMPLETE event
            # For now, simulate a brief pause and return to IDLE
            await asyncio.sleep(0.5)
            
            # Restart audio capture before going back to IDLE
            await audio_source.start()
            
            await fsm.transition(State.IDLE)
            logger.info("System returned to IDLE state. Ready for next command.")
            
        except Exception as e:
            logger.error(f"Error during LLM processing: {e}", exc_info=True)
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
        asyncio.create_task(llm_engine.cancel())
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
                
                # Feed STT and VAD if LISTENING
                elif fsm.state == State.LISTENING:
                    await stt_engine.feed_audio(frame)
                    
                    if vad_engine:
                        is_currently_speaking = vad_engine.is_speech(frame)
                        
                        # Detect transition from speaking to silent
                        if was_speaking and not is_currently_speaking:
                            logger.info("Silence detected. Stopping STT stream.")
                            await stt_engine.stop_stream()
                        
                        was_speaking = is_currently_speaking
                
                # In other states (THINKING, SPEAKING), we still consume the frame
                # but do nothing with it to prevent queue overflow.
                else:
                    pass
            else:
                await asyncio.sleep(0.01)
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
    finally:
        await audio_source.stop()
        await wakeword_engine.stop()
        await llm_engine.cancel()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
