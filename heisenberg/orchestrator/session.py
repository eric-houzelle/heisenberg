import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Tuple

@dataclass
class Session:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=datetime.utcnow)
    conversation_history: List[Tuple[str, str]] = field(default_factory=list)  # (user_query, assistant_response)
    
    def fresh_correlation_id(self) -> str:
        self.correlation_id = str(uuid.uuid4())
        return self.correlation_id
    
    def add_turn(self, user_query: str, assistant_response: str):
        """Add a conversation turn to history."""
        self.conversation_history.append((user_query, assistant_response))
    
    def get_history(self, max_turns: int = None) -> List[Tuple[str, str]]:
        """
        Get conversation history, optionally limited to most recent turns.
        
        Args:
            max_turns: Maximum number of turns to return (most recent). None = all.
            
        Returns:
            List of (user_query, assistant_response) tuples
        """
        if max_turns is None:
            return self.conversation_history.copy()
        return self.conversation_history[-max_turns:]
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()

class SessionManager:
    def __init__(self):
        self._current_session: Optional[Session] = None

    def start_new_session(self) -> Session:
        self._current_session = Session()
        return self._current_session

    @property
    def current_session(self) -> Optional[Session]:
        return self._current_session
    
    def add_conversation_turn(self, user_query: str, assistant_response: str):
        """Add a conversation turn to the current session."""
        if self._current_session:
            self._current_session.add_turn(user_query, assistant_response)
    
    def get_conversation_history(self, max_turns: int = None) -> List[Tuple[str, str]]:
        """Get conversation history from current session."""
        if self._current_session:
            return self._current_session.get_history(max_turns)
        return []
    
    def clear_conversation_history(self):
        """Clear conversation history in current session."""
        if self._current_session:
            self._current_session.clear_history()
