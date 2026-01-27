from torch.utils.data import DataLoader, random_split
import dataset
import torch.optim as optim
from network import Net
import torch
import torch.nn as nn
from collections import Counter

# Load dataset with sample limit for testing
data = dataset.Dataset("groove/", max_samples=50000)

# Check class distribution
label_counts = Counter([label.item() for label in data.labels])
print("\n" + "="*50)
print("CLASS DISTRIBUTION:")
print("="*50)
for class_idx in range(dataset.NUM_CLASSES):
    count = label_counts.get(class_idx, 0)
    percentage = 100 * count / len(data.labels)
    print(f"{dataset.CLASS_NAMES[class_idx]:12s}: {count:5d} samples ({percentage:5.2f}%)")
print("="*50)

# Check data statistics
sample_spec = data.specs[0]
print(f"\nSample spectrogram shape: {sample_spec.shape}")
print(f"Sample spectrogram min: {sample_spec.min():.4f}")
print(f"Sample spectrogram max: {sample_spec.max():.4f}")
print(f"Sample spectrogram mean: {sample_spec.mean():.4f}")
print(f"Sample spectrogram std: {sample_spec.std():.4f}")

train_size = int(0.8 * data.__len__())
test_size = data.__len__() - train_size

print(f"\nTrain size: {train_size}")
print(f"Test size: {test_size}")
print("Splitting data...")
train_data, test_data = random_split(data, [train_size, test_size])

batch_size = 32

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")

model = Net(num_classes=11).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

print(f"\nModel architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

num_epochs = 50

print(f"\nStarting training for {num_epochs} epochs...")
print("="*70)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Debug first batch
        if epoch == 0 and i == 0:
            print(f"\nFirst batch info:")
            print(f"  Input shape: {inputs.shape}")
            print(f"  Input min: {inputs.min():.4f}, max: {inputs.max():.4f}")
            print(f"  Input mean: {inputs.mean():.4f}, std: {inputs.std():.4f}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Unique labels in batch: {labels.unique().tolist()}")

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_acc = 100 * correct / total
    avg_loss = running_loss / len(train_loader)
    
    print(f'Epoch [{epoch+1:2d}/{num_epochs}] | Loss: {avg_loss:.4f} | Train Acc: {train_acc:5.2f}%', end='')
    
    # Validation every 5 epochs
    if (epoch + 1) % 5 == 0:
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        avg_val_loss = val_loss / len(test_loader)
        print(f' | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:5.2f}%')
    else:
        print()

print("="*70)
print('Training complete!')

torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'drum_classifier.pth')
print('Model saved!')