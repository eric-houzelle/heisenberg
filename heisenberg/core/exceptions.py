class HeisenbergError(Exception):
    """Base exception for all application errors."""
    pass

class AudioError(HeisenbergError):
    pass

class WakeWordError(HeisenbergError):
    pass

class STTError(HeisenbergError):
    pass

class LLMError(HeisenbergError):
    pass

class TTSError(HeisenbergError):
    pass

class ConfigurationError(HeisenbergError):
    pass
