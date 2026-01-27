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

# TODO : 
# - Now normalize per window, not globally.
# - S_dB = (S_dB - np.mean(S_dB)) / (np.std(S_dB) + 1e-6)
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

    for i, (time, note) in enumerate(midi_onsets):
        # Position onset closer to the start of the window
        onset_sample = int(time * sr)
        pre_onset_samples = int(frame_length * pre_onset_ratio)
        
        # Add jitter for data augmentation (±5ms random shift)
        #jitter = np.random.randint(-int(0.005*sr), int(0.005*sr))
        
        start_sample = onset_sample - pre_onset_samples # + jitter
        end_sample = start_sample + frame_length
        
        # Handle edge cases (beginning of audio)
        if start_sample < 0:
            start_sample = 0
            end_sample = frame_length
        if end_sample > len(y):
            end_sample = len(y)
            start_sample = max(0, end_sample - frame_length)
        
        y_window = y[start_sample:end_sample]

        # helper.plot_waveform(y_window, sr, pre_onset_samples)

        n_fft = 512
        # For center=False: time_bins = floor((n_samples - n_fft) / hop_length) + 1
        # We want 18 time bins, so: hop_length = (n_samples - n_fft) / 17
        hop_length = (len(y_window) - n_fft) // 17
        
        #print(f"Window length: {len(y_window)} samples")
        #print(f"Calculated hop_length: {hop_length}")
        #print(f"Expected time bins: {(len(y_window) - n_fft) // hop_length + 1}")

        spec = helper.compute_spectrogram(y_window, sr, plot=False, n_mels=128, n_fft=n_fft, hop_length=hop_length)
        #spec_image = helper.spec_to_image(spec)
        spec_image = (spec - spec.mean()) / (spec.std() + 1e-6)


        label = drum_mapping.get(note, None)
        if label is None:
            continue
        spectrograms.append(spec_image)
        labels.append(label)

    return spectrograms, labels