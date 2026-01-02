import pytest
from heisenberg.orchestrator.fsm import FSM
from heisenberg.orchestrator.state import State
from heisenberg.orchestrator.events import Event
from heisenberg.orchestrator.router import EventRouter

@pytest.mark.asyncio
async def test_fsm_initial_state():
    router = EventRouter()
    fsm = FSM(router)
    await fsm.start()
    assert fsm.state == State.IDLE

@pytest.mark.asyncio
async def test_fsm_wakeword_transition():
    router = EventRouter()
    fsm = FSM(router)
    await fsm.start()
    
    await fsm.handle_event(Event.WAKEWORD_DETECTED)
    assert fsm.state == State.LISTENING
