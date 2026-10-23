"""
Oja's rule × PCA
"""

import torch
from sklearn.decomposition import PCA


class Oja:
    """Online estimator of the first principal component."""

    def __init__(self, n_features: int, lr: float = 1e-2, seed: int = 0):
        torch.manual_seed(seed)
        self.lr = lr
        self.w = torch.randn(n_features, dtype=torch.float32)
        self.w /= self.w.norm()

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Apply one Oja update: dw = lr * y * (x - y * w)."""
        y = torch.dot(self.w, x)
        self.w += self.lr * y * (x - y * self.w)
        self.w /= self.w.norm()

    @torch.no_grad()
    def fit(
            self,
            X: torch.Tensor,
            epochs: int = 10,
            shuffle: bool = True,
    ) -> torch.Tensor:
        """Train on X with shape [n_samples, n_features]."""
        n_samples = X.shape[0]

        for _ in range(epochs):
            indices = (
                torch.randperm(n_samples)
                if shuffle
                else torch.arange(n_samples)
            )

            for i in indices:
                self.update(X[i])

        return self.w

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Project X onto the learned 1D component."""
        return X @ self.w


def pca_first_component(X: torch.Tensor) -> torch.Tensor:
    """Return PCA's first component as a unit vector."""
    pca = PCA(n_components=1)
    pca.fit(X.numpy())

    w = torch.from_numpy(pca.components_[0]).to(dtype=X.dtype)
    return w / w.norm()


def component_alignment(
        w1: torch.Tensor,
        w2: torch.Tensor,
) -> float:
    """
    Absolute cosine similarity between two components.

    The absolute value is important because w and -w
    represent the same principal one-dimensional subspace.
    """
    return abs(torch.dot(w1, w2)).item()


def make_data(
        n_samples: int = 2000,
        seed: int = 0,
) -> torch.Tensor:
    """Create centered 2D data elongated approximately along x2 = x1."""
    torch.manual_seed(seed)

    z = torch.randn(n_samples, 1)
    noise = 0.25 * torch.randn(n_samples, 2)

    direction = torch.tensor([[1.0, 1.0]]) / torch.sqrt(
        torch.tensor(2.0)
    )

    X = z @ direction + noise
    X -= X.mean(dim=0, keepdim=True)

    return X


def main() -> None:
    X = make_data()

    oja = Oja(
        n_features=X.shape[1],
        lr=1e-2,
        seed=1,
    )

    w_oja = oja.fit(
        X,
        epochs=20,
        shuffle=True,
    )

    w_pca = pca_first_component(X)

    alignment = component_alignment(w_oja, w_pca)

    print("Oja component:")
    print(w_oja)

    print("\nPCA component:")
    print(w_pca)

    print("\nAbsolute cosine similarity:")
    print(alignment)

    z_oja = oja.transform(X)
    z_pca = X @ w_pca

    print("\nProjection shapes:")
    print("Oja:", z_oja.shape)
    print("PCA:", z_pca.shape)


if __name__ == "__main__":
    main()
