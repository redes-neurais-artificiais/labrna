import torch
from labrna.dataset import mnist
from labrna.mlp import MLP
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

X, y = mnist(2000)
X = X[y < 2]
y = y[y < 2]

model = MLP(64)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

X_tensor = torch.FloatTensor(X)
y = torch.FloatTensor(y).unsqueeze(1)
dataset = TensorDataset(X_tensor, y)
dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

num_epochs = 100
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for data, target in dataloader:
        # Forward
        yhat = model(data)
        loss = criterion(yhat, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if (epoch + 1) % 2 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}')
