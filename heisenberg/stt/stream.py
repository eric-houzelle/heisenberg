from heisenberg.interfaces.stt import ABCSTT
from typing import Callable, Awaitable

class STTStream(ABCSTT):
    async def start_stream(self) -> None:
        pass

    async def stop_stream(self) -> None:
        pass

    async def feed_audio(self, frame: bytes) -> None:
        pass

    def on_partial(self, callback: Callable[[str], Awaitable[None]]) -> None:
        pass

    def on_final(self, callback: Callable[[str], Awaitable[None]]) -> None:
        pass
