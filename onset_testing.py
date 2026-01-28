import librosa as lb
import helper
import dataset
from mido import MidiFile
import numpy as np

def get_all_onsets(file_path, sr=22050, midi=False):
    """Extract all onset times from audio or MIDI file."""
    if midi is False:
        # Load audio
        audio, sr = lb.load(file_path + "wav", sr=sr)
        
        # General onset detection with maximum sensitivity
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
        onset_frames = [t + offset for t in onset_frames]
        return onset_frames
    else:
        # Extract MIDI onsets
        midi_file = MidiFile(file_path + "mid")
        current_time = 0
        unproc_frames = []
        
        for msg in midi_file:
            current_time += msg.time
            if (msg.type == 'note_on' and msg.velocity > 0):
                if (msg.note in dataset.DRUM_MAPPING):
                    unproc_frames.append(current_time)
        
        # Group close onsets
        group_tolerance = 0.01
        onset_frames = []
        current_group_time = None
        
        for onset in unproc_frames:
            if current_group_time is None:
                current_group_time = onset
            else:
                if (onset - current_group_time) <= group_tolerance:
                    continue
                else:
                    onset_frames.append(current_group_time)
                    current_group_time = onset
        
        if current_group_time is not None:
            onset_frames.append(current_group_time)
        
        return onset_frames


def compare_onsets(file_path, midi_idx, offset=0, sr=22050):
    """Compare MIDI vs librosa onsets and plot them."""
    import matplotlib.pyplot as plt
    
    # Get all onsets
    midi_onsets = get_all_onsets(file_path, sr=sr, offset=offset, midi=True)
    librosa_onsets = get_all_onsets(file_path, sr=sr, offset=offset, midi=False)
    
    print(f"\n🎯 Target MIDI onset [{midi_idx}]: {midi_onsets[midi_idx]:.4f}s")
    print(f"[MIDI] {len(midi_onsets)} onsets")
    print(f"[Librosa] {len(librosa_onsets)} onsets")
    
    # Find closest librosa onset
    target_midi_onset = midi_onsets[midi_idx]
    closest_idx = min(range(len(librosa_onsets)), 
                     key=lambda i: abs(librosa_onsets[i] - target_midi_onset))
    librosa_onset_time = librosa_onsets[closest_idx]
    
    time_diff = abs(target_midi_onset - librosa_onset_time)
    print(f"🔍 Closest Librosa onset [{closest_idx}]: {librosa_onset_time:.4f}s")
    print(f"   Time difference: {time_diff*1000:.2f}ms\n")
    
    # Load audio and extract windows
    audio, sr = lb.load(file_path + "wav", sr=sr, duration=10, offset=offset)
    frame_length = int(0.1 * sr)
    pre_onset_ratio = 0.3
    pre_onset_samples = int(frame_length * pre_onset_ratio)
    
    # Extract MIDI window
    frame_in_audio = target_midi_onset - offset
    onset_sample = int(frame_in_audio * sr)
    start = max(0, onset_sample - pre_onset_samples)
    end = min(len(audio), start + frame_length)
    if end - start < frame_length:
        start = max(0, end - frame_length)
    midi_window = audio[start:end]
    
    # Extract librosa window
    frame_in_audio = librosa_onset_time - offset
    onset_sample = int(frame_in_audio * sr)
    start = max(0, onset_sample - pre_onset_samples)
    end = min(len(audio), start + frame_length)
    if end - start < frame_length:
        start = max(0, end - frame_length)
    librosa_window = audio[start:end]
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    midi_start_time = target_midi_onset - (pre_onset_samples / sr)
    time_axis_midi = midi_start_time + lb.samples_to_time(range(len(midi_window)), sr=sr)
    
    librosa_start_time = librosa_onset_time - (pre_onset_samples / sr)
    time_axis_librosa = librosa_start_time + lb.samples_to_time(range(len(librosa_window)), sr=sr)
    
    ax.plot(time_axis_midi, midi_window, color='blue', alpha=0.6, linewidth=1.5, label='MIDI Window')
    ax.plot(time_axis_librosa, librosa_window, color='green', alpha=0.6, linewidth=1.5, label='Librosa Window')
    
    ax.axvline(x=target_midi_onset, color='red', linestyle='--', alpha=0.8, linewidth=2, 
               label=f'MIDI Onset ({target_midi_onset:.3f}s)')
    ax.axvline(x=librosa_onset_time, color='orange', linestyle='--', alpha=0.8, linewidth=2, 
               label=f'Librosa Onset ({librosa_onset_time:.3f}s)')
    
    ax.set_title('Onset Detection Comparison: MIDI vs Librosa (Multi-Method)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    print(f"Onset Comparison:")
    print(f"  MIDI onset:    {target_midi_onset:.4f}s")
    print(f"  Librosa onset: {librosa_onset_time:.4f}s")
    print(f"  Difference:    {time_diff*1000:.2f}ms")
    if time_diff < 0.01:
        print(f"  ✓ Onsets match within 10ms tolerance")
    else:
        print(f"  ✗ Onsets differ by more than 10ms")
    
    plt.tight_layout()
    plt.show()
