import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from labrna.dataset import mnist

# Load MNIST data
X, y = mnist(1000)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X_scaled)
dataset = TensorDataset(X_tensor, X_tensor)  # Autoencoder uses input as target
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# Define a simple autoencoder
class Autoencoder(nn.Module):
    """
    A simple autoencoder model with an encoder and decoder.
    """
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64, 60),
            nn.ReLU(),
            nn.Linear(60, 30),
            nn.ReLU(),
            nn.Linear(30, 2)  # Bottleneck layer - 2D for visualization
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 30),
            nn.ReLU(),
            nn.Linear(30, 60),
            nn.ReLU(),
            nn.Linear(60, 64)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# Initialize model, loss and optimizer
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 64
for epoch in range(num_epochs):
    for data, target in dataloader:
        # Forward pass
        encoded, decoded = model(data)
        loss = criterion(decoded, target)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Get the 2D encodings for visualization
with torch.no_grad():
    encoded_imgs, _ = model(X_tensor)
    embeddings = encoded_imgs.numpy()

plt.figure(figsize=(10, 8))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(label='Digit Class')
plt.title('Autoencoder 2D Embedding of MNIST Sample')
plt.show()

# Plot the results
pca = PCA()
embeddings = pca.fit_transform(X_scaled)
# Plot the results
plt.figure(figsize=(10, 8))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(label='Digit Class')
plt.title('PCA 2D Embedding of MNIST Sample')
plt.show()

tsne = TSNE()
embeddings = tsne.fit_transform(X_scaled)
# Plot the results
plt.figure(figsize=(10, 8))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(label='Digit Class')
plt.title('TSNE 2D Embedding of MNIST Sample')
plt.show()
