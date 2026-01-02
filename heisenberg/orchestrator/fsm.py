import logging
import asyncio
from typing import Optional

from heisenberg.orchestrator.state import State
from heisenberg.orchestrator.events import Event
from heisenberg.orchestrator.router import EventRouter
from heisenberg.orchestrator.policies import Policies
from heisenberg.orchestrator.session import SessionManager

logger = logging.getLogger(__name__)

class FSM:
    def __init__(self, router: EventRouter, policies: Policies = None):
        self.state = State.IDLE
        self.router = router
        self.policies = policies or Policies()
        self.session_manager = SessionManager()

    async def transition(self, new_state: State):
        if self.state == new_state:
            return

        old_state = self.state
        self.state = new_state
        logger.info(f"FSM Transition: {old_state.name} -> {new_state.name}", 
                    extra={"old_state": old_state.name, "new_state": new_state.name})

        # Hook for state entry logic if needed (or dispatch a STATE_CHANGED event)

    async def handle_event(self, event: Event, *args, **kwargs):
        """
        Main entry point for events into the FSM. 
        Validation of transitions can happen here.
        """
        # Example validation: simpler for now, just dispatch
        if event == Event.WAKEWORD_DETECTED:
            if self.state == State.IDLE: # Only valid from IDLE? Or barge-in?
                await self.transition(State.LISTENING)
        
        elif event == Event.TRANSCRIPTION_FINAL:
             if self.state == State.LISTENING:
                 await self.transition(State.THINKING)
        
        elif event == Event.LLM_TOKEN and self.state == State.THINKING:
             # First token signifies we might start speaking or buffering
             pass 
             
        elif event == Event.TTS_START:
             await self.transition(State.SPEAKING)

        elif event == Event.TTS_COMPLETE:
             await self.transition(State.IDLE)
        
        # Dispatch to registered handlers to do the actual work
        await self.router.dispatch(event, *args, **kwargs)

    async def start(self):
        logger.info("FSM Started")
        self.session_manager.start_new_session()
        await self.transition(State.IDLE)
