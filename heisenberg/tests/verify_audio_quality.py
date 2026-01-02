import asyncio
import wave
import numpy as np
from heisenberg.core.config import Config
from heisenberg.audio.capture import PyAudioIO

async def record_sample(duration=5, filename="test_capture.wav"):
    config = Config.load()
    audio = PyAudioIO(config.audio)
    
    print(f"Recording {duration} seconds to {filename}...")
    print("Speak 'Hey Jarvis' during the recording.")
    
    await audio.start()
    
    frames = []
    num_frames = int(duration * 16000 / config.audio.chunk_size)
    
    for _ in range(num_frames):
        frame = await audio.read_frame()
        if frame:
            frames.append(frame)
        await asyncio.sleep(0.01)
        
    await audio.stop()
    
    # Save to WAV
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2) # 16-bit
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))
    
    print(f"Done! Please listen to {filename} and check if the audio is clear and not distorted.")

if __name__ == "__main__":
    asyncio.run(record_sample())
