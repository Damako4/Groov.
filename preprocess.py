import librosa as lb
import os
import csv
import matplotlib.pyplot as plt
from mido import MidiFile
import numpy as np
import helper
from tqdm import tqdm

# PARAMETERS TO MODIFY
# - window_size: 200 ms (0.2 s)
# - sample_rate: 44100 Hz
# - grouping_tolerance: 10 ms
# - pre_onset_ratio: 30%
# - STFT settings

# TODO : Optimization steps
# - Volume variation
# - Negative samples (windows with no onsets)

sample_rate = 22050 # Half to 22050 Hz
    
# Preprocess data
def get_audio_data(path, drum_mapping):
    y, sr = lb.load(path + "wav", sr=sample_rate)
    #print("Loading: ", path + ".wav")
    midi_onsets = []
    
    # Load MIDI
    midi = MidiFile(path + "mid")
    current_time = 0
    for msg in midi:
        current_time += msg.time  # Accumulate delta times
        if (msg.type == 'note_on' and msg.velocity > 0):
            if (msg.note in drum_mapping):
                midi_onsets.append([current_time, msg.note]) # Ignoring velocity as an input parameter

    # Group close onsets
    group_tolerance = 0.01 # 10ms
    processed_onsets = []
    current_group = None # (onset, [notes])
    for onset, note in midi_onsets:
        if not current_group:
            current_group = (onset, [note])
        else:
            if (onset - current_group[0] <= group_tolerance):
                current_group[1].append(note)
            else:
                processed_onsets.append(current_group)  # Choose earliest onset
                current_group = (onset, [note])  # Start new group
    if current_group:
        processed_onsets.append(current_group)

    # Window parameters
    frame_length = int(0.1 * sr)  # 100ms total window
    pre_onset_ratio = 0.3  # 30% before onset, 70% after
    hop_length = frame_length // 2
    
    spectrograms = []
    labels = []

    # Use processed_onsets (grouped simultaneous hits) instead of midi_onsets
    positive_windows = []
    window_centers = []
    for i, (time, notes) in enumerate(processed_onsets):
        onset_sample = int(time * sr)
        pre_onset_samples = int(frame_length * pre_onset_ratio)
        jitter = np.random.randint(-int(0.005*sr), int(0.005*sr))
        start_sample = onset_sample - pre_onset_samples + jitter
        end_sample = start_sample + frame_length
        if start_sample < 0:
            start_sample = 0
            end_sample = frame_length
        if end_sample > len(y):
            end_sample = len(y)
            start_sample = max(0, end_sample - frame_length)
        y_window = y[start_sample:end_sample]

        # --- Simple augmentations ---
        aug_windows = [y_window]
        # Gain (random between 0.7x and 1.3x)
        gain = np.random.uniform(0.7, 1.3)
        aug_windows.append(y_window * gain)
        # Additive Gaussian noise (std 2% of signal std)
        noise = np.random.normal(0, 0.02 * np.std(y_window), size=y_window.shape)
        aug_windows.append(y_window + noise)
        # Pitch shift (random between -1 and 1 semitones)
        n_steps = np.random.uniform(-1, 1)
        aug_windows.append(lb.effects.pitch_shift(y_window, sr, n_steps=n_steps))
        # Time stretch (±10%)
        stretch = np.random.uniform(0.9, 1.1)
        try:
            aug_windows.append(lb.effects.time_stretch(y_window, rate=stretch))
        except Exception:
            pass  # If too short for stretch, skip

        label_vector = np.zeros(11, dtype=np.float32)
        for note in notes:
            label_idx = drum_mapping.get(note, None)
            if label_idx is not None:
                label_vector[label_idx] = 1.0

        for aug_win in aug_windows:
            # Pad/crop to frame_length
            if len(aug_win) < frame_length:
                aug_win = np.pad(aug_win, (0, frame_length - len(aug_win)))
            elif len(aug_win) > frame_length:
                aug_win = aug_win[:frame_length]
            n_fft = 512
            hop_length = (len(aug_win) - n_fft) // 17
            spec = helper.compute_spectrogram(aug_win, sr, plot=False, n_mels=128, n_fft=n_fft, hop_length=hop_length)
            spec_image = (spec - spec.mean()) / (spec.std() + 1e-6)
            if label_vector.sum() > 0:
                spectrograms.append(spec_image)
                labels.append(label_vector)
                positive_windows.append((start_sample, end_sample))
                window_centers.append((start_sample + end_sample) // 2)

    # Add negative samples: 1 per positive sample
    n_neg = len(positive_windows)
    min_gap = int(0.05 * sr)  # 50ms gap from any positive window center
    for _ in range(n_neg):
        for _ in range(20):  # Try up to 20 times to find a valid negative
            center = np.random.randint(frame_length//2, len(y) - frame_length//2)
            # Check if this window overlaps any positive window (center at least min_gap away)
            if all(abs(center - c) > (frame_length//2 + min_gap) for c in window_centers):
                start_sample = center - frame_length//2
                end_sample = start_sample + frame_length
                y_window = y[start_sample:end_sample]
                n_fft = 512
                hop_length = (len(y_window) - n_fft) // 17
                spec = helper.compute_spectrogram(y_window, sr, plot=False, n_mels=128, n_fft=n_fft, hop_length=hop_length)
                spec_image = (spec - spec.mean()) / (spec.std() + 1e-6)
                label_vector = np.zeros(11, dtype=np.float32)  # All zeros = no drum
                spectrograms.append(spec_image)
                labels.append(label_vector)
                break

    return spectrograms, labels