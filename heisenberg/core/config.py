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
    threshold: float = 0.5
    inference_framework: str = "onnxrt" # or "tflite"

@dataclass
class Config:
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    wakeword: WakewordConfig = field(default_factory=WakewordConfig)
    
    @classmethod
    def load(cls) -> "Config":
        # TODO: Load from env or file
        return cls()
