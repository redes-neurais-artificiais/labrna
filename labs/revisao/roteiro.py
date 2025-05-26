"""
Conceitos de Redes Neurais com o dataset Wine Quality (Red)
Vinhos com qualidade >= 6 são considerados bons (1), os demais ruins (0).
"""

import torch
import torch.nn as nn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("""\n\n=====================================================
Carregando o dataset...
-----------------------------------------------------""")
data = fetch_openml(name='wine-quality-red', version=1, as_frame=True)
print("ok!")
X = data.data.values
y = data.target.astype(int).values
y = (y >= 6).astype(int)  # binarização

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

print("""\n\n=====================================================
Questão 1: Espaço de Instâncias e Espaço de Hipóteses
Defina uma hipótese linear arbitrária e teste em 5 instâncias do conjunto de treinamento.
-----------------------------------------------------""")
w = torch.randn(X_train.shape[1])
b = torch.randn(1).item()


def h(x_):
    return torch.heaviside(torch.dot(x_, w) + b, torch.tensor([0.0]))


for i in range(5):
    print(f"Instância {i}: pred = {h(X_train[i])}, rótulo real = {y_train[i].item()}")

print("""\n\n=====================================================
Questão 2: O problema é linearmente separável?
Treine um Perceptron e verifique a acurácia.
-----------------------------------------------------""")
w = torch.zeros(X_train.shape[1])
b = 0.0
for epoch in range(10):
    for xi, yi in zip(X_train, y_train):
        pred = torch.heaviside(torch.dot(xi, w) + b, torch.tensor([0.0]))
        w += 0.01 * (yi.item() - pred.item()) * xi
        b += 0.01 * (yi.item() - pred.item())

with torch.no_grad():
    preds = torch.heaviside(torch.matmul(X_test, w) + b, torch.tensor([0.0]))
    acc = (preds.unsqueeze(1) == y_test).float().mean()
    print(f"Acurácia com Perceptron: {acc:.4f}")

print("""\n\n=====================================================
Questão 3: Mostre o vetor normal e o intercepto do hiperplano linear aprendido.
-----------------------------------------------------""")
print(f"Vetor w: {[round(float(v), 3) for v in w]}")
print(f"Intercepto b: {b}")

print("""\n\n=====================================================
Questão 4: Adicione atributos (aleatórios) e observe o impacto da dimensionalidade.
-----------------------------------------------------""")
for extra_dims in [0, 4, 16, 64, 256, 1024]:
    X_aug = torch.cat([X_train, torch.randn(X_train.shape[0], extra_dims)], dim=1)
    model = nn.Sequential(nn.Linear(X_aug.shape[1], 10), nn.ReLU(), nn.Linear(10, 1), nn.Sigmoid())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()
    for _ in range(20):
        y_pred = model(X_aug)
        loss = loss_fn(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        X_test_aug = torch.cat([X_test, torch.randn(X_test.shape[0], extra_dims)], dim=1)
        acc = ((model(X_test_aug) > 0.5) == y_test).float().mean()
        print(f"Extra dims: {extra_dims}, Acurácia: {acc:.4f}")

print("""\n\n=====================================================
Questão 5: MLP como Aproximador Universal.
Treine MLPs com diferentes tamanhos de camada oculta e observe a acurácia.
-----------------------------------------------------""")
for hidden_size in [4, 8, 16, 32]:
    model = nn.Sequential(nn.Linear(X_train.shape[1], hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1), nn.Sigmoid())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()
    for _ in range(50):
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        acc = ((model(X_test) > 0.5) == y_test).float().mean()
        print(f"Hidden size: {hidden_size}, Acurácia: {acc:.4f}")

print("""\n\n=====================================================
Questão 6: MLP Autoassociativa para redução de dimensionalidade
Treine um autoencoder e projete os dados na camada escondida.
-----------------------------------------------------""")


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        z = F.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z


auto = Autoencoder(X_train.shape[1], hidden_dim=3)
optimizer = torch.optim.Adam(auto.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for _ in range(100):
    x_hat, _ = auto(X_train)
    loss = loss_fn(x_hat, X_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    _, embeddings = auto(X_test)
    print("Exemplo de embedding das primeiras 5 instâncias (3D):")
    print(embeddings[:5])
