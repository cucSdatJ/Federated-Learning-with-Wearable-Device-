import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int = 8, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.net(x)