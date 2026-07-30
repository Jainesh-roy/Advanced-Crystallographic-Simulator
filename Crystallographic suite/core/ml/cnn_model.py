"""
cnn_model.py

1D Convolutional Neural Network for XRD pattern classification.
"""

import torch
import torch.nn as nn


class XRDClassifier(nn.Module):
    """
    Input Shape:
        (Batch, 1, 3501)

    Output:
        (Batch, num_classes)
    """

    def __init__(self, num_classes):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=7,
                padding=3
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                32,
                64,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                64,
                128,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


if __name__ == "__main__":

    model = XRDClassifier(num_classes=5)

    x = torch.randn(4, 1, 3501)

    y = model(x)

    print(model)

    print("\nInput :", x.shape)

    print("Output:", y.shape)