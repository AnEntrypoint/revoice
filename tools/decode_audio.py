import sys
import wave

from pydub import AudioSegment

audio = AudioSegment.from_file(sys.argv[1])
audio = audio.set_channels(1)
audio = audio.set_sample_width(2)

out_path = sys.argv[2]
with wave.open(out_path, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(audio.frame_rate)
    w.writeframes(audio.raw_data)

print(f"sample_rate={audio.frame_rate}")
print(f"duration_s={len(audio) / 1000.0}")
print(f"num_samples={len(audio.raw_data) // 2}")
