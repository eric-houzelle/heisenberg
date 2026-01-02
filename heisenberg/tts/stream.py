from heisenberg.interfaces.tts import ABCTTS

class TTSStream(ABCTTS):
    async def speak(self, text_chunk: str) -> None:
        pass

    async def stop(self) -> None:
        pass
