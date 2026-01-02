# Heisenberg - Modular Voice Assistant

Heisenberg is a modular, high-performance voice assistant built with Python. It features real-time wakeword detection, Voice Activity Detection (VAD), and Speech-to-Text (STT) capabilities, orchestrated through a Finite State Machine (FSM).

---

## 🏗️ Architecture

Heisenberg follows a decoupled, event-driven architecture. Components communicate asynchronously via an `EventRouter`, allowing for high responsiveness and easy swapability of modules.

```mermaid
graph TD
    A[PyAudioIO] -->|Raw Audio| B(Audio Pipeline)
    B -->|RNNoise/Resample/AGC| C{Event Router}
    C -->|16kHz Chunk| D[OpenWakeWordEngine]
    C -->|16kHz Chunk| E[SileroVADEngine]
    C -->|16kHz Chunk| F[WhisperSTT]
    
    D -->|WAKEWORD_DETECTED| G[FSM]
    E -->|SPEECH_START/END| G
    F -->|TRANSCRIPTION_FINAL| G
    
    G -->|Transition| H[State: IDLE | LISTENING | THINKING]
```

### Core Components
- **Audio Layer (`heisenberg.audio`)**: Real-time capture uses PyAudio.
- **Wakeword Layer (`heisenberg.wakeword`)**: Uses `openwakeword` for background listening.
- **STT Layer (`heisenberg.stt`)**: Leverages `pywhispercpp` (GGML models) for local, fast transcription.
- **Orchestrator (`heisenberg.orchestrator`)**: Manages transitions and business logic via an FSM.

---

## �️ System Requirements

Before installing Python dependencies, ensure your system has the following:

- **OS**: Linux (Optimized for Ubuntu/Debian), macOS, or Windows.
- **Python**: 3.12 or higher.
- **C++ Build Tools**: Required for compiling `pywhispercpp` and `pyaudio` bindings.
    - Ubuntu: `sudo apt install build-essential python3-dev`
- **PortAudio**: Required for `pyaudio`.
    - Ubuntu: `sudo apt install libportaudio2 libasound2-dev`
    - macOS: `brew install portaudio`

---

## 🚀 Installation & Setup

### 1. Requirements Installation
Using `uv` (recommended) or `pip`:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and enter the app directory
git clone https://github.com/your-repo/heisenberg.git
cd heisenberg/app

# Initialize environment and install dependencies
uv sync
```

### 2. Model Downloads
Whisper models are loaded locally. By default, it expects a model file (e.g., `base-q8_0.bin`) in your working directory or a path specified in the config.

---

## ⚙️ Configuration Reference

Edit `heisenberg/core/config.py` to customize the behavior.

### Audio Config (`AudioConfig`)
| Field | Default | Description |
| :--- | :--- | :--- |
| `input_device_index` | `-1` | Index of the microphone (use `-1` for default). |
| `sample_rate` | `16000` | Target sample rate for processing (fixed at 16k). |
| `channels` | `1` | Number of audio channels (Mono required). |
| `chunk_size` | `1280` | Size of the audio buffer chunks. |

### Wakeword Config (`WakewordConfig`)
| Field | Default | Description |
| :--- | :--- | :--- |
| `models` | `["hey_jarvis"]` | List of `openwakeword` models to load. |
| `threshold` | `0.3` | Sensitivity threshold for activation (0.0 to 1.0). |

### STT Config (`STTConfig`)
| Field | Default | Description |
| :--- | :--- | :--- |
| `model_path` | `"base-q8_0"` | Path to the `.bin` GGML whisper model. |
| `language` | `"fr"` | Transcription language (ISO 639-1). |
| `n_threads` | `4` | Number of CPU threads for Whisper inference. |
| `debug_dump` | `True` | Dumps the last recorded audio to `.wav` for quality check. |

---

## 🎙️ Audio Pipeline Deep Dive

Heisenberg implements a sophisticated audio pipeline in `PyAudioIO` to ensure high quality even in noisy environments:

1.  **High-Res Capture**: If `RNNoise` is available, audio is captured at **48kHz**.
2.  **Denoising**: `RNNoise` (Recurrent Neural Network for Noise Suppression) processes audio in 10ms chunks to remove background hum and steady noise.
3.  **Resampling**: The signal is downsampled to **16kHz** (the standard for Whisper and Wakeword engines).
4.  **Automatic Gain Control (AGC)**: Simple RMS-based normalization ensures the signal isn't too quiet or clipping before inference.

---

## 🏃 Running the Assistant

Execute the main loop:
```bash
uv run heisenberg/main.py
```

### Debugging
If you encounter audio issues:
1.  Check `debug_stt_*.wav` files generated in the project root to listen to what the assistant actually heard.
2.  Verify your microphone index by listing devices (tools coming soon).
3.  Set `logging.level = "DEBUG"` in `config.py` for verbose output.

---

## 🧪 Testing

Run the test suite to ensure everything is wired correctly:
```bash
uv run pytest
```

---

## 📜 License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
