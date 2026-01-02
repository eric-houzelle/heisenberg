from abc import ABC, abstractmethod
from typing import Optional

class ABCAudioIO(ABC):
    """
    Interface for low-level Audio Input/Output.
    Responsibility: Read microphone input and write speaker output.
    """

    @abstractmethod
    async def read_frame(self) -> Optional[bytes]:
        """Read a frame of audio from the input device."""
        pass

    @abstractmethod
    async def play_frame(self, frame: bytes) -> None:
        """Write a frame of audio to the output device."""
        pass
        
    @abstractmethod
    async def start(self) -> None:
        """Start audio streams."""
        pass
        
    @abstractmethod
    async def stop(self) -> None:
        """Stop audio streams."""
        pass
