import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, num_classes):
        # 64 feature maps
        # Collapse time to 12 and frequency to 16 => (16, 12, 64)
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding="same"), # (128, 18, 32)
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)), # (64, 9, 32)
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding="same"), # (64, 9, 64)
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)), # (32, 4, 64)
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding="same"), # (32, 4, 64)
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)) # (16, 2, 64)
        )

        self.fc_stack = nn.Sequential(
            nn.Linear(16 * 2 * 64, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        x = self.fc_stack(x)
        return x
