import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 기본 Residual Block 정의
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, stride=1):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            identity = self.downsample(x)

        return self.relu(out + identity)
    
# 전체 Appearance CNN 정의
class DeepSortEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),   # Conv1
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),  # Conv2
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),       # MaxPool3

            ResidualBlock(32),  # Res4
            ResidualBlock(32),  # Res5

            ResidualBlock(32, 64, stride=2),  # Res6 (downsample)
            ResidualBlock(64),                # Res7

            ResidualBlock(64, 128, stride=2), # Res8 (downsample)
            ResidualBlock(128),               # Res9
        )

        self.embedding = nn.Sequential(
            nn.Flatten(),                    # flatten to (128×16×8) = 16384
            nn.Linear(128 * 16 * 8, 128),    # Dense 10
            nn.BatchNorm1d(128),
        )

    def forward(self, x):
        x = self.net(x)                    # CNN block
        x = self.embedding(x)             # FC layer
        x = F.normalize(x, p=2, dim=1)    # L2 normalization to unit hypersphere
        return x                          # shape: (N, 128)