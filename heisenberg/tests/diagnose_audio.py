import pyaudio

def list_audio_devices():
    pa = pyaudio.PyAudio()
    print("\nAvailable Audio Input Devices:")
    print("-" * 30)
    
    default_input = pa.get_default_input_device_info()
    print(f"Default Input Device: {default_input['name']} (Index: {default_input['index']})")
    
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"Index {i}: {info['name']}")
            print(f"  Max Channels: {info['maxInputChannels']}")
            print(f"  Default Sample Rate: {info['defaultSampleRate']}")
            
            # Test common rates
            supported_rates = []
            for rate in [16000, 44100, 48000]:
                try:
                    if pa.is_format_supported(rate, input_device=i, input_channels=1, input_format=pyaudio.paInt16):
                        supported_rates.append(rate)
                except:
                    pass
            print(f"  Supported Rates Tested: {supported_rates}")
    
    pa.terminate()

if __name__ == "__main__":
    list_audio_devices()
