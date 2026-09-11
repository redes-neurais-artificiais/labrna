import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features, 256), nn.Sigmoid(),
            nn.Linear(256, 128), nn.Sigmoid(),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
