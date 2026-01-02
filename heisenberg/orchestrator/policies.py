from dataclasses import dataclass, field

@dataclass
class Timeouts:
    wakeword_listen: float = 0.0 # 0 = infinite
    stt_silence: float = 2.0
    llm_generation: float = 30.0
    tts_playback: float = 60.0

@dataclass
class Policies:
    timeouts: Timeouts = field(default_factory=Timeouts)
    allow_barge_in: bool = True
    
    # Retry policies could go here
    max_retries: int = 3
