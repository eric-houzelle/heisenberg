import json
import logging
import asyncio
from typing import AsyncGenerator, Optional, Callable
import aiohttp

from heisenberg.interfaces.llm import ABCLLM
from heisenberg.core.config import LLMConfig
from heisenberg.llm.prompts import PromptBuilder

logger = logging.getLogger(__name__)

class LlamaCppLLM(ABCLLM):
    """
    LLM client for llama.cpp HTTP server.
    Supports streaming token generation.
    """
    
    def __init__(self, config: LLMConfig, prompt_builder: Optional[PromptBuilder] = None):
        """
        Args:
            config: LLM configuration
            prompt_builder: Optional prompt builder (will create default if None)
        """
        self.config = config
        self.prompt_builder = prompt_builder or PromptBuilder(
            system_prompt=config.system_prompt,
            format_style="plain"  # Adjust based on your model
        )
        self._current_session: Optional[aiohttp.ClientSession] = None
        self._current_task: Optional[asyncio.Task] = None
        self._on_token_callback: Optional[Callable[[str], None]] = None
        self._on_complete_callback: Optional[Callable[[str], None]] = None
    
    def on_token(self, callback: Callable[[str], None]):
        """Register callback for each token generated."""
        self._on_token_callback = callback
    
    def on_complete(self, callback: Callable[[str], None]):
        """Register callback when generation is complete."""
        self._on_complete_callback = callback
    
    async def generate(self, prompt: str, conversation_history: list = None) -> AsyncGenerator[str, None]:
        """
        Generate text response from a prompt with streaming.
        
        Args:
            prompt: The user's query
            conversation_history: Optional list of (user, assistant) tuples for context
            
        Yields:
            Individual tokens as they are generated
        """
        conversation_history = conversation_history or []
        
        # Build complete prompt with history
        full_prompt = self.prompt_builder.build(conversation_history, prompt)
        
        logger.debug(f"Sending prompt to LLM (length: {len(full_prompt)} chars)")
        
        # Prepare request payload for llama.cpp
        payload = {
            "prompt": full_prompt,
            "temperature": self.config.temperature,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
            "n_predict": self.config.max_tokens,
            "repeat_penalty": self.config.repeat_penalty,
            "stop": ["User:", "user:", "<|im_end|>", "</s>"],  # Stop sequences
            "stream": True,
        }
        
        full_response = ""
        token_count = 0
        first_token = True
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                self._current_session = session
                
                async with session.post(self.config.endpoint, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"LLM API error {response.status}: {error_text}")
                        raise RuntimeError(f"LLM request failed: {response.status}")
                    
                    logger.info("Started receiving LLM stream")
                    
                    # Parse SSE stream from llama.cpp
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        
                        if not line or line.startswith(':'):
                            continue
                        
                        # llama.cpp sends "data: " prefix for SSE
                        if line.startswith('data: '):
                            data_str = line[6:]  # Remove "data: " prefix
                            
                            try:
                                data = json.loads(data_str)
                                
                                # Extract token content
                                token = data.get('content', '')
                                
                                if token:
                                    full_response += token
                                    token_count += 1
                                    
                                    # Log first token for latency tracking
                                    if first_token:
                                        logger.info("Received first LLM token")
                                        first_token = False
                                    
                                    # Call token callback if registered
                                    if self._on_token_callback:
                                        self._on_token_callback(token)
                                    
                                    yield token
                                
                                # Check if generation is complete
                                if data.get('stop', False):
                                    logger.info(f"LLM generation complete. Tokens: {token_count}")
                                    break
                                    
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse LLM response: {e}, line: {data_str}")
                                continue
        
        except asyncio.CancelledError:
            logger.info("LLM generation cancelled")
            raise
        
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}", exc_info=True)
            raise
        
        finally:
            self._current_session = None
            
            # Call completion callback
            if self._on_complete_callback:
                self._on_complete_callback(full_response)
        
        logger.debug(f"Generated response: {full_response[:100]}...")
    
    async def generate_simple(self, prompt: str, conversation_history: list = None) -> str:
        """
        Generate complete response (non-streaming convenience method).
        
        Args:
            prompt: The user's query
            conversation_history: Optional conversation context
            
        Returns:
            Complete generated text
        """
        full_response = ""
        async for token in self.generate(prompt, conversation_history):
            full_response += token
        return full_response
    
    async def cancel(self) -> None:
        """Cancel the current generation."""
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            logger.info("Cancelled LLM generation task")
        
        if self._current_session and not self._current_session.closed:
            await self._current_session.close()
            logger.info("Closed LLM session")


class LLMStream(ABCLLM):
    """Deprecated: Use LlamaCppLLM instead."""
    
    async def generate(self, prompt: str) -> AsyncGenerator[str, None]:
        yield ""

    async def cancel(self) -> None:
        pass
