from abc import ABC, abstractmethod
from typing import Callable, Awaitable

class ABCWakeword(ABC):
    """
    Interface for Wakeword detection engines.
    Responsibility: Detect a specific keyword in the audio stream.
    """
    
    @abstractmethod
    async def start(self) -> None:
        """Start the wakeword detection engine."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the wakeword detection engine."""
        pass

    @abstractmethod
    def on_detected(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Register a callback to be called when the wakeword is detected."""
        pass
