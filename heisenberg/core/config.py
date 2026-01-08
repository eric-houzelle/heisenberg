from dataclasses import dataclass, field

@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "json"

@dataclass
class AudioConfig:
    input_device_index: int = -1
    output_device_index: int = -1
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1280

@dataclass
class WakewordConfig:
    models: list[str] = field(default_factory=lambda: ["hey_jarvis"])
    threshold: float = 0.1
    inference_framework: str = "onnxrt"

@dataclass
class STTConfig:
    model_path: str = "base-q8_0"
    language: str = "fr"
    n_threads: int = 4
    sampling_strategy: int = 1 # 0: GREEDY, 1: BEAM_SEARCH
    initial_prompt: str = "Bonjour, je suis ton assistant Heisenberg."
    debug_dump: bool = True # Enable dumping audio to WAV for quality check

@dataclass
class VADConfig:
    enabled: bool = True
    threshold: float = 0.5
    min_silence_duration_ms: int = 800
    speech_pad_ms: int = 100

@dataclass
class LLMConfig:
    endpoint: str = "http://localhost:8080/completion"
    model_name: str = "LFM2-350M"
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    timeout_seconds: int = 30
    system_prompt: str = "Tu es Heisenberg, un assistant vocal intelligent et serviable. Réponds de manière concise et naturelle."
    max_history_turns: int = 5  # Number of conversation turns to keep in context

@dataclass
class Config:
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    wakeword: WakewordConfig = field(default_factory=WakewordConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    
    @classmethod
    def load(cls) -> "Config":
        # TODO: Load from env or file
        return cls()
