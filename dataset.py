from torch.utils.data import Dataset
import torch
import helper
import preprocess
import tqdm
import pickle
import os

# MIDI pitch to drum class mapping
DRUM_MAPPING = {
    36: 0,   # Kick
    38: 1,   # Snare (Head)
    40: 1,   # Snare (Rim) -> Snare
    37: 1,   # Snare X-Stick -> Snare
    48: 2,   # Tom 1 (Head)
    45: 3,   # Tom 2 (Head)
    43: 4,   # Tom 3 (Head)
    46: 5,   # HH Open (Bow)
    26: 5,   # HH Open (Edge) -> HH Open
    42: 6,   # HH Closed (Bow)
    22: 6,   # HH Closed (Edge) -> HH Closed
    44: 7,   # HH Pedal
    49: 8,   # Crash 1 (Bow)
    55: 8,   # Crash 1 (Edge) -> Crash
    57: 8,   # Crash 2 (Bow) -> Crash
    52: 8,   # Crash 2 (Edge) -> Crash
    51: 9,   # Ride (Bow)
    59: 9,   # Ride (Edge) -> Ride
    53: 10,  # Ride (Bell)
}

CLASS_NAMES = [
    "Kick",           # 0
    "Snare",          # 1
    "Tom 1",          # 2
    "Tom 2",          # 3
    "Tom 3",          # 4
    "HH Open",        # 5
    "HH Closed",      # 6
    "HH Pedal",       # 7
    "Crash",          # 8
    "Ride",           # 9
    "Ride Bell",      # 10
]

NUM_CLASSES = len(CLASS_NAMES)  # 11 classes 

class Dataset(Dataset):
    def __init__(self, path, cache_file="dataset_cache.pkl", force_reprocess=False):
        self.specs = []
        self.labels = []
        self.cache_file = cache_file

        # Try to load from cache
        if os.path.exists(cache_file) and not force_reprocess:
            print(f"Loading dataset from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.specs = cache_data['specs']
                self.labels = cache_data['labels']
            print(f"✓ Loaded {len(self.specs)} samples from cache")
        else:
            if force_reprocess:
                print("Force reprocessing dataset...")
            else:
                print("No cache found. Loading dataset from scratch...")
            
            for path in tqdm.tqdm(helper.walk_fast(path)):
                specs, labels = preprocess.get_audio_data(path, DRUM_MAPPING)
                # Extend lists with all spectrograms from this file
                for spec, label in zip(specs, labels):
                    self.specs.append(torch.tensor(spec, dtype=torch.float32).unsqueeze(0))
                    self.labels.append(torch.tensor(label, dtype=torch.long))
            
            # Save to cache
            print(f"Saving dataset to cache: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'specs': self.specs,
                    'labels': self.labels
                }, f)
            print(f"✓ Cached {len(self.specs)} samples")

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        return self.specs[idx], self.labels[idx]