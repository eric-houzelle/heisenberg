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
    threshold: float = 0.3
    inference_framework: str = "onnxrt"

@dataclass
class STTConfig:
    model_path: str = "tiny-q8_0"
    language: str = "fr"
    n_threads: int = 4
    debug_dump: bool = True # Enable dumping audio to WAV for quality check

@dataclass
class Config:
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    wakeword: WakewordConfig = field(default_factory=WakewordConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    
    @classmethod
    def load(cls) -> "Config":
        # TODO: Load from env or file
        return cls()
