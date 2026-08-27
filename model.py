import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """Small CNN for MFCC-based audio deepfake detection.

    Input: (batch, 40, time_steps) - 40 MFCC coefficients with channel dim, variable time steps
    Output: raw logits with shape (batch, 1) representing P(FAKE)
    """

    def __init__(self, n_mfcc=40, n_classes=1):
        super(SmallCNN, self).__init__()

        # Expects input with shape (batch, 40, time_steps)
        # Add conv layer that accepts 40 input channels (the MFCC coefficients)
        self.features = nn.Sequential(
            # Block 1: (batch, 1, 40, time_steps) -> (batch, 32, 40//2, time_steps//2)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2: (batch, 32, ..., ...) -> (batch, 64, ..., ...)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Adaptive pooling to handle variable time steps
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        # x shape should be (batch, 40, time_steps)
        # Add channel dimension for Conv2d: (batch, 1, 40, time_steps)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 40, time_steps) -> (batch, 1, 40, time_steps)

        x = self.features(x)
        # x shape: (batch, 64, reduced_h, reduced_w)
        x = self.global_pool(x)
        # x shape: (batch, 64, 1, 1)
        x = self.classifier(x)
        # Output shape: (batch, 1) - raw logits
        return x