"""
Test module for LLM integration.
Run this to verify your llama.cpp server is configured correctly.
"""

import asyncio
import logging
from heisenberg.core.config import Config
from heisenberg.llm.stream import LlamaCppLLM
from heisenberg.llm.prompts import PromptBuilder, SYSTEM_PROMPTS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_simple_query():
    """Test a simple LLM query without history."""
    config = Config.load()
    
    # Create prompt builder with concise style for testing
    prompt_builder = PromptBuilder(
        system_prompt=SYSTEM_PROMPTS["concise"],
        format_style="plain"
    )
    
    llm = LlamaCppLLM(config.llm, prompt_builder)
    
    query = "Quelle est la capitale de la France ?"
    
    logger.info(f"Sending query: {query}")
    logger.info("Waiting for LLM response...")
    
    try:
        response = await llm.generate_simple(query)
        logger.info(f"Response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


async def test_streaming_query():
    """Test streaming token generation."""
    config = Config.load()
    llm = LlamaCppLLM(config.llm)
    
    query = "Explique-moi en quelques mots ce qu'est l'intelligence artificielle."
    
    logger.info(f"Sending query: {query}")
    logger.info("Streaming response:")
    
    print("\n--- Response ---")
    try:
        async for token in llm.generate(query):
            print(token, end='', flush=True)
        print("\n--- End ---\n")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


async def test_conversation_with_history():
    """Test conversation with context."""
    config = Config.load()
    llm = LlamaCppLLM(config.llm)
    
    # Simulate conversation history
    history = [
        ("Comment t'appelles-tu ?", "Je m'appelle Heisenberg."),
        ("Quelle est ta fonction ?", "Je suis un assistant vocal intelligent."),
    ]
    
    query = "Rappelle-moi ton nom."
    
    logger.info(f"Conversation history: {len(history)} turns")
    logger.info(f"Current query: {query}")
    
    try:
        response = await llm.generate_simple(query, conversation_history=history)
        logger.info(f"Response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


async def main():
    logger.info("=" * 60)
    logger.info("LLM Integration Test Suite")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Make sure llama.cpp server is running!")
    logger.info("Example: ./llama-server -m models/lfm2-350m-q8_0.gguf -c 2048")
    logger.info("")
    logger.info("=" * 60)
    
    await asyncio.sleep(1)
    
    # Test 1: Simple query
    logger.info("\n[TEST 1] Simple Query")
    logger.info("-" * 60)
    try:
        await test_simple_query()
        logger.info("✓ Test 1 passed")
    except Exception as e:
        logger.error(f"✗ Test 1 failed: {e}")
    
    await asyncio.sleep(2)
    
    # Test 2: Streaming
    logger.info("\n[TEST 2] Streaming Query")
    logger.info("-" * 60)
    try:
        await test_streaming_query()
        logger.info("✓ Test 2 passed")
    except Exception as e:
        logger.error(f"✗ Test 2 failed: {e}")
    
    await asyncio.sleep(2)
    
    # Test 3: Conversation history
    logger.info("\n[TEST 3] Conversation with History")
    logger.info("-" * 60)
    try:
        await test_conversation_with_history()
        logger.info("✓ Test 3 passed")
    except Exception as e:
        logger.error(f"✗ Test 3 failed: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Test suite complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())


