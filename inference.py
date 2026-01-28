import torch
from network import Net
import dataset
import helper
import preprocess
import numpy as np
import onset

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Net(num_classes=11).to(device)
checkpoint = torch.load("drum_classifier.pth", weights_only=False, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load and process audio
hits = onset.get_all_onsets("groove/drummer5/session1/16_rock_136_beat_4-4.", sr=22050, duration=10)

if len(hits) == 0:
    print("No onsets detected!")
    exit()

# Inference
with torch.no_grad():
    for hit_spec, hit_time in hits:
        spec_tensor = torch.tensor(hit_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        output = model(spec_tensor)
        predictions = (torch.sigmoid(output) > 0.5).float()

        # Print which drums were detected
        detected = False
        drums_detected = []
        for i, pred in enumerate(predictions[0]):
            if pred > 0:
                drums_detected.append(dataset.CLASS_NAMES[i])
                detected = True
        
        if detected:
            print(f"[{hit_time:.3f}s] Detected: {', '.join(drums_detected)}")
        # Uncomment below to see all onsets including empty ones
        # else:
        #     print(f"[{hit_time:.3f}s] No drums detected")