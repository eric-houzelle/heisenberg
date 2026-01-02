from abc import ABC, abstractmethod
from typing import Callable, Awaitable

class ABCSTT(ABC):
    """
    Interface for Speech-to-Text engines.
    Responsibility: Transform audio stream into text.
    """

    @abstractmethod
    async def start_stream(self) -> None:
        """Start the STT streaming session."""
        pass

    @abstractmethod
    async def stop_stream(self) -> None:
        """Stop the STT streaming session."""
        pass

    @abstractmethod
    async def feed_audio(self, frame: bytes) -> None:
        """Feed audio data to the STT engine."""
        pass

    @abstractmethod
    def on_partial(self, callback: Callable[[str], Awaitable[None]]) -> None:
        """Register callback for partial transcription updates."""
        pass

    @abstractmethod
    def on_final(self, callback: Callable[[str], Awaitable[None]]) -> None:
        """Register callback for final transcription results."""
        pass
