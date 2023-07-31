import wave
import numpy as np
import matplotlib.pyplot as plt

signal_wave = wave.open(
    "/scratch/gpfs/kw1166/whisper-decoder/data/podcast/Podcast.wav", "r"
)
sample_rate = 16000
sig = np.frombuffer(signal_wave.readframes(sample_rate), dtype=np.int16)
sig = sig[:10000]

fig, ax = plt.subplots()

ax.plot(sig)
# ax.set_xlabel('sample rate * time')
# ax.set_ylabel('energy')
ax.set_ylim(-20000, 20000)

# plot_b = plt.subplot(212)
# plot_b.specgram(sig, NFFT=1024, Fs=sample_rate, noverlap=900)
# plot_b.set_xlabel('Time')
# plot_b.set_ylabel('Frequency')

plt.savefig(f"audio.png")
