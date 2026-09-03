import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """
    CNN for MFCC-based deepfake audio detection.

    Input:
        (batch, 1, 40, time_steps)

    Output:
        Raw logit with shape (batch, 1)
    """

    def __init__(self, n_classes=1):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.10),

            # Block 2
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.15),

            # Block 3
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.20),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),

            nn.Linear(64, n_classes)
        )

    def forward(self, x):

        # Dataset returns:
        # (batch, 1, 40, time_steps)
        #
        # If input is:
        # (batch, 40, time_steps)
        # add channel dimension.

        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.features(x)

        x = self.global_pool(x)

        x = self.classifier(x)

        return x