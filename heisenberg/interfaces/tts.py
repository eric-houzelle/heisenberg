from abc import ABC, abstractmethod

class ABCTTS(ABC):
    """
    Interface for Text-to-Speech engines.
    Responsibility: Synthesize speech from text.
    """

    @abstractmethod
    async def speak(self, text_chunk: str) -> None:
        """
        Synthesize and play speech for the given text chunk.
        Ideally non-blocking or managed via internal queue.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        Immediately stop speech playback and clear queues.
        """
        pass
