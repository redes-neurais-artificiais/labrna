import torch
from labrna.dataset import mnist
from labrna.mlp import MLP
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

X, y = mnist(100)
model = MLP(64)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

X_tensor = torch.FloatTensor(X)
dataset = TensorDataset(X_tensor, X_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

num_epochs = 10
for epoch in range(num_epochs):
    for data, target in dataloader:
        # Forward
        ...

        # Backward
        ...

    if (epoch + 1) % 2 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
