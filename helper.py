import librosa as lb
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_waveform(y, sr, pre_onset_samples):
    fig, ax = plt.subplots()
    # Create normalized time axis with onset at pre_onset_ratio position
    time_axis = lb.samples_to_time(range(len(y)), sr=sr) - (pre_onset_samples / sr)
    
    # Plot window
    ax.plot(time_axis, y)
    
    # Mark the onset with a vertical line at x=0
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.7, linewidth=2, label='Onset')

    ax.legend()
    plt.show()

def compute_spectrogram(y, sr, plot=False, n_mels=128, hop_length=512, n_fft=512):
    S = lb.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=8000, hop_length=hop_length, n_fft=n_fft, center=False)
    S_dB = lb.power_to_db(S, ref=np.max)
    #print(f"Spectrogram shape: {S_dB.shape}")  # Debug print
    if plot:
        fig, ax = plt.subplots()
        img = lb.display.specshow(S_dB, x_axis='time',
                            y_axis='mel', sr=sr,
                            fmax=8000, ax=ax, hop_length=hop_length)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='Mel-frequency spectrogram')
        plt.show()
    return S_dB

def spec_to_image(spec, eps=1e-6):
  mean = spec.mean()
  std = spec.std()
  spec_norm = (spec - mean) / (std + eps)
  spec_min, spec_max = spec_norm.min(), spec_norm.max()
  spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
  spec_scaled = spec_scaled.astype(np.uint8)
  return spec_scaled


# Get filepath independent of WAV or MIDI
def walk_fast(path):
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_dir(follow_symlinks=False):
                yield from walk_fast(entry.path)
            else:
                if (entry.path[-2] == 'a'):
                    yield entry.path[:-3]    