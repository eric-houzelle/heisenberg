import logging
from typing import Callable, Dict, Awaitable, Any

from heisenberg.orchestrator.events import Event
from heisenberg.core.logging import get_correlation_id

logger = logging.getLogger(__name__)

EventHandler = Callable[..., Awaitable[Any]]

class EventRouter:
    def __init__(self):
        self._handlers: Dict[Event, EventHandler] = {}

    def register(self, event: Event, handler: EventHandler):
        self._handlers[event] = handler

    async def dispatch(self, event: Event, *args, **kwargs):
        handler = self._handlers.get(event)
        if handler:
            try:
                logger.debug(f"Dispatching event {event.name}", extra={"event": event.name})
                await handler(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error handling event {event.name}: {e}", exc_info=True)
                # In a real app, we might want to re-raise or trigger an error event
        else:
            logger.warning(f"No handler for event {event.name}")
