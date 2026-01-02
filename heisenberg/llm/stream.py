from heisenberg.interfaces.llm import ABCLLM
from typing import AsyncGenerator

class LLMStream(ABCLLM):
    async def generate(self, prompt: str) -> AsyncGenerator[str, None]:
        yield ""

    async def cancel(self) -> None:
        pass
