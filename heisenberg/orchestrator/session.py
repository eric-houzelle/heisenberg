import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class Session:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=datetime.utcnow)
    
    def fresh_correlation_id(self) -> str:
        self.correlation_id = str(uuid.uuid4())
        return self.correlation_id

class SessionManager:
    def __init__(self):
        self._current_session: Optional[Session] = None

    def start_new_session(self) -> Session:
        self._current_session = Session()
        return self._current_session

    @property
    def current_session(self) -> Optional[Session]:
        return self._current_session
