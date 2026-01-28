import librosa as lb
import helper
import dataset
from mido import MidiFile
import numpy as np

def get_all_onsets(file_path, sr=22050, midi=False, duration=10):
    """Extract all onset times from audio file using librosa onset detection."""
    audio, sr = lb.load(file_path + "wav", sr=sr, duration=duration)
    onset_frames_idx = lb.onset.onset_detect(
        y=audio, sr=sr, hop_length=128,
        backtrack=True,
        normalize=False,
        pre_max=1,
        post_max=1,
        pre_avg=3,
        post_avg=3,
        delta=0.02,
        wait=5,
        units='frames',
    )
    onset_frames = lb.frames_to_time(onset_frames_idx, sr=sr, hop_length=128)

    # Window parameters
    frame_length = int(0.1 * sr)  # 100ms window
    pre_onset_ratio = 0.3

    hits = []
    for time in onset_frames:
        onset_sample = int(time * sr)
        pre_onset_samples = int(frame_length * pre_onset_ratio)
        start_sample = onset_sample - pre_onset_samples
        end_sample = start_sample + frame_length

        # Edge cases
        if start_sample < 0:
            start_sample = 0
            end_sample = frame_length
        if end_sample > len(audio):
            end_sample = len(audio)
            start_sample = max(0, end_sample - frame_length)

        y_window = audio[start_sample:end_sample]
        
        n_fft = 512
        # For center=False: time_bins = floor((n_samples - n_fft) / hop_length) + 1
        # We want 18 time bins, so: hop_length = (n_samples - n_fft) / 17
        hop_length = (len(y_window) - n_fft) // 17

        spec = helper.compute_spectrogram(y_window, sr, plot=False, n_mels=128, n_fft=n_fft, hop_length=hop_length)
        spec_image = (spec - spec.mean()) / (spec.std() + 1e-6)
        
        hits.append((spec_image, time))


    return hits

