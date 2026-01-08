from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class Message:
    """Represents a single message in a conversation."""
    role: str  # "system", "user", or "assistant"
    content: str

class PromptBuilder:
    """
    Helper for constructing LLM prompts with conversation history.
    Supports multiple prompt formats (ChatML, Llama, etc.)
    """
    
    def __init__(self, system_prompt: str = "", format_style: str = "chatml"):
        """
        Args:
            system_prompt: The system instruction to prepend to conversations
            format_style: Format style for prompts ("chatml", "llama2", "plain")
        """
        self.system_prompt = system_prompt
        self.format_style = format_style
    
    def build(self, history: List[Tuple[str, str]], current_query: str) -> str:
        """
        Build a complete prompt from conversation history.
        
        Args:
            history: List of (user_message, assistant_response) tuples
            current_query: The current user query to respond to
            
        Returns:
            Formatted prompt string ready for LLM inference
        """
        messages = []
        
        # Add system prompt if present
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))
        
        # Add conversation history
        for user_msg, assistant_msg in history:
            messages.append(Message(role="user", content=user_msg))
            messages.append(Message(role="assistant", content=assistant_msg))
        
        # Add current query
        messages.append(Message(role="user", content=current_query))
        
        # Format according to style
        if self.format_style == "chatml":
            return self._format_chatml(messages)
        elif self.format_style == "llama2":
            return self._format_llama2(messages)
        else:
            return self._format_plain(messages)
    
    def _format_chatml(self, messages: List[Message]) -> str:
        """Format messages in ChatML style (used by many modern models)."""
        formatted = []
        for msg in messages:
            formatted.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")
        formatted.append("<|im_start|>assistant\n")
        return "\n".join(formatted)
    
    def _format_llama2(self, messages: List[Message]) -> str:
        """Format messages in Llama 2 chat style."""
        # Extract system message
        system_msg = ""
        dialog_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                dialog_messages.append(msg)
        
        # Build Llama 2 format
        if system_msg:
            prompt = f"[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n"
        else:
            prompt = "[INST] "
        
        for i, msg in enumerate(dialog_messages):
            if msg.role == "user":
                if i > 0:
                    prompt += "[INST] "
                prompt += f"{msg.content} [/INST]"
            else:  # assistant
                prompt += f" {msg.content} </s><s>"
        
        return prompt
    
    def _format_plain(self, messages: List[Message]) -> str:
        """Format messages in plain text style."""
        formatted = []
        for msg in messages:
            if msg.role == "system":
                formatted.append(f"System: {msg.content}")
            elif msg.role == "user":
                formatted.append(f"User: {msg.content}")
            else:
                formatted.append(f"Assistant: {msg.content}")
        formatted.append("Assistant:")
        return "\n\n".join(formatted)


# Predefined system prompts for different personalities
SYSTEM_PROMPTS = {
    "default": "Tu es Heisenberg, un assistant vocal intelligent et serviable. Réponds de manière concise et naturelle en français.",
    
    "concise": "Tu es un assistant vocal. Réponds en 1-2 phrases maximum. Sois direct et précis.",
    
    "friendly": "Tu es Heisenberg, un assistant vocal chaleureux et amical. Tu utilises un ton décontracté et tu aimes aider. Réponds de manière naturelle et conversationnelle.",
    
    "professional": "Tu es un assistant professionnel. Réponds de manière claire, structurée et courtoise. Fournis des informations précises.",
    
    "technical": "Tu es un assistant technique spécialisé. Fournis des réponses détaillées avec des explications techniques quand c'est approprié. Utilise un vocabulaire précis.",
}
