import pytest
from heisenberg.orchestrator.events import Event
from heisenberg.orchestrator.router import EventRouter

@pytest.mark.asyncio
async def test_event_router_dispatch():
    router = EventRouter()
    
    received_event = None
    
    async def handler():
        nonlocal received_event
        received_event = True

    router.register(Event.SPEECH_START, handler)
    await router.dispatch(Event.SPEECH_START)
    
    assert received_event is True
