"""
LLM Module - Heisenberg

Integration of local language models via llama.cpp.

Main Components:
- LlamaCppLLM: Asynchronous LLM client with streaming support
- PromptBuilder: Prompt construction with conversation history
- SYSTEM_PROMPTS: Predefined personality templates

Example Usage:
    from heisenberg.llm import LlamaCppLLM, PromptBuilder
    from heisenberg.core.config import Config
    
    config = Config.load()
    llm = LlamaCppLLM(config.llm)
    
    # Simple generation
    response = await llm.generate_simple("Hello")
    
    # Streaming generation
    async for token in llm.generate("Tell me a joke"):
        print(token, end='', flush=True)
"""

from heisenberg.llm.stream import LlamaCppLLM
from heisenberg.llm.prompts import PromptBuilder, SYSTEM_PROMPTS

__all__ = [
    "LlamaCppLLM",
    "PromptBuilder",
    "SYSTEM_PROMPTS",
]


