from enum import Enum, auto

class Event(Enum):
    WAKEWORD_DETECTED = auto()
    SPEECH_START = auto()
    SPEECH_END = auto()
    TRANSCRIPTION_FINAL = auto()
    LLM_TOKEN = auto()
    LLM_COMPLETE = auto()
    TTS_START = auto()
    TTS_COMPLETE = auto()
    ERROR_OCCURRED = auto()
    TIMEOUT = auto()
    INTERRUPT = auto() # Barge-in
