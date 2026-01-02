from abc import ABC, abstractmethod
from typing import AsyncGenerator

class ABCLLM(ABC):
    """
    Interface for Large Language Models.
    Responsibility: Generate text tokens from a prompt.
    """

    @abstractmethod
    async def generate(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Generate text response from a prompt.
        Returns an async generator yielding tokens.
        """
        pass

    @abstractmethod
    async def cancel(self) -> None:
        """Cancel the current generation."""
        pass
