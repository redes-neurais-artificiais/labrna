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

    show_visual_demo(X, w_oja, w_pca)



def show_visual_demo(
        X: torch.Tensor,
        w_oja: torch.Tensor,
        w_pca: torch.Tensor,
        lr: float = 1e-2,
        epochs: int = 20,
        seed: int = 1,
) -> None:
    """Open a window showing data, components, convergence and projections."""
    import math

    import matplotlib.pyplot as plt
    import numpy as np

    # Replay Oja's rule update by update to record how its angle to PCA shrinks.
    oja = Oja(n_features=X.shape[1], lr=lr, seed=seed)
    angles = []
    for _ in range(epochs):
        for i in torch.randperm(X.shape[0]):
            oja.update(X[i])
            cos = min(1.0, abs(float(torch.dot(oja.w, w_pca))))
            angles.append(math.degrees(math.acos(cos)))

    data = X.numpy()
    z_oja = (X @ w_oja).numpy()
    z_pca = (X @ w_pca).numpy()
    w_oja_np = w_oja.numpy()
    w_pca_np = w_pca.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Oja's rule converging to the first principal component", fontsize=14)

    # 1) Data cloud with both directions drawn through the origin.
    ax = axes[0]
    ax.scatter(data[:, 0], data[:, 1], s=6, alpha=0.3, color="gray", label="samples")
    t = np.array([-3.0, 3.0])
    ax.plot(t * w_pca_np[0], t * w_pca_np[1], "r-", lw=3, label="PCA (sklearn)")
    ax.plot(t * w_oja_np[0], t * w_oja_np[1], "b--", lw=2, label="Oja (final)")
    ax.set_aspect("equal")
    ax.set_title("Data and principal direction")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="upper left")

    # 2) Angle between Oja's w and PCA's component across updates.
    ax = axes[1]
    ax.plot(angles, color="blue")
    ax.set_yscale("log")
    ax.set_title("Angle between Oja's w and PCA (log scale)")
    ax.set_xlabel("update step")
    ax.set_ylabel("degrees")
    ax.grid(alpha=0.3)

    # 3) 1D projections: both methods should give (up to sign) the same spread.
    ax = axes[2]
    ax.hist(z_pca, bins=60, color="red", alpha=0.5, label="PCA projection")
    ax.hist(z_oja, bins=60, color="blue", alpha=0.5, label="Oja projection")
    ax.set_title("1D projections onto the component")
    ax.set_xlabel("z")
    ax.set_ylabel("count")
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
