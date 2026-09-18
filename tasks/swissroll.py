import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import NearestNeighbors

# Gerar rocambole
n_samples = 3000
X, _ = make_swiss_roll(n_samples)
X = torch.tensor(X, dtype=torch.float32)

# Mantém apenas pontos até completar o comprimento.
comprimento = 15
mask = (X[:, 1] > 0) & (X[:, 1] < comprimento)
X = X[mask]

# Visualização da variedade
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='b', s=1)
ax.set_xlim(-12, 12)
ax.set_ylim(0, 24)
ax.set_zlim(-12, 12)
plt.title("Rocambole (variedade 2D imersa em R³)")


# Estimador simples de dimensionalidade intrínseca baseado em PCA local
def estimate_intrinsic_dim(X, k=10):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, indices = nbrs.kneighbors(X)

    local_dims = []
    for idx in indices:
        neighbors = X[idx[1:]]  # ignorar o próprio ponto
        neighbors -= neighbors.mean(dim=0)
        cov = neighbors.T @ neighbors / k
        eigvals = torch.linalg.eigvalsh(cov)
        eigvals = eigvals.flip(0)
        eigvals /= eigvals.sum()
        dim = (eigvals ** 2).sum().reciprocal()  # Neff (entropia de Rényi de ordem 2)
        local_dims.append(dim.item())
    return local_dims


# Estima dimensionalidade local
dims = estimate_intrinsic_dim(X, k=30)

fig = plt.figure(figsize=(5, 5))
plt.hist(dims, bins=30)
plt.xlabel("Dimensionalidade intrínseca estimada")
plt.ylabel("Frequência")
plt.show()

# Média e desvio-padrão
mean_dim = np.mean(dims)
std_dim = np.std(dims)
print(f"Dimensionalidade intrínseca média: {mean_dim:.2f} ± {std_dim:.2f}")

# Seja moderado(a) na fatia do rocambole e veja o efeito na dimensionalidade intrínseca estimada.
