# espacos
import torch
import torch.nn as nn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.nn import Linear, ReLU, Sequential
from torch.optim import SGD
from torch.utils.data import TensorDataset, DataLoader


def instancia_e_hipotese():
    instances = [(torch.tensor([0.0, 0.0]), 0),
                 (torch.tensor([0.0, 1.0]), 1),
                 (torch.tensor([1.0, 0.0]), 1),
                 (torch.tensor([1.0, 1.0]), 0)]

    def hypothesis(x, w, b):
        return torch.heaviside(torch.dot(w, x) + b, torch.tensor([0.0]))

    return instances, hypothesis


# portas_logicas
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y_and = torch.tensor([0., 0., 0., 1.])
y_xor = torch.tensor([0., 1., 1., 0.])

# hiperplano
w = torch.tensor([1.0, -1.0])
b = 0.0


def hyperplane(x):
    return torch.dot(w, x) + b


# mal_dimensionalidade
d = 1000
n = 100
X_hd = torch.randn(n, d)
avg_dist = torch.cdist(X_hd, X_hd).mean()


# erro_e_minimos
def mse(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()


def loss_fn(x):
    return torch.sin(5 * x) * torch.exp(-x ** 2)


# generalizacao
muitos_parametros = Sequential(Linear(1, 1000), ReLU(), Linear(1000, 1))

# perceptron_componentes
x = torch.tensor([1.0, 0.0])
w = torch.tensor([0.5, -0.5])
b = 0.1
potencial = torch.dot(w, x) + b
saida = torch.heaviside(potencial, torch.tensor([0.0]))


# perceptron_treinamento
def perceptron_train(X, y, lr=0.1, epochs=10):
    w = torch.zeros(X.shape[1])
    b = 0.0
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            pred = torch.heaviside(torch.dot(w, xi) + b, torch.tensor([0.0]))
            w += lr * (yi - pred) * xi
            b += lr * (yi - pred)
    return w, b


# mlp_aproximador
mlp = nn.Sequential(nn.Linear(2, 10), nn.Tanh(), nn.Linear(10, 1))

# ativacoes
x = torch.linspace(-5, 5, 100)
linear = x
step = torch.heaviside(x, torch.tensor([0.0]))
sigmoid = torch.sigmoid(x)
tanh = torch.tanh(x)
relu = torch.relu(x)

# xor_resolucao
xor_net = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1), nn.Sigmoid())

# gradiente_descendente
w = torch.randn(1, requires_grad=True)
for _ in range(100):
    loss = (w - 3) ** 2
    loss.backward()
    with torch.no_grad():
        w -= 0.1 * w.grad
        w.grad.zero_()

# backprop
model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1), nn.Sigmoid())
criterion = nn.BCELoss()
optimizer = SGD(model.parameters(), lr=0.1)

for _ in range(1000):
    y_pred = model(X)
    loss = criterion(y_pred.squeeze(), y_and)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# funcoes_erro
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()

# manifold
theta = torch.linspace(0, 4 * torch.pi, 100)
x = theta * torch.cos(theta)
y = theta * torch.sin(theta)
z = theta
spiral = torch.stack([x, y, z], dim=1)


# autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(10, 3)
        self.decoder = nn.Linear(3, 10)

    def forward(self, x):
        z = torch.relu(self.encoder(x))
        return self.decoder(z)


def carregar_dataset_wine_quality(batch_size=32):
    data = fetch_openml(name='wine-quality-red', version=1, as_frame=True)
    X = data.data.values
    y = data.target.astype(int).values  # classes de 3 a 8

    # Reduzir para problema binário não trivial: qualidade >=6 como bom (1), caso contrário ruim (0)
    y = (y >= 6).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Conversão para tensores
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    return train_loader, test_loader, X_train.shape[1]
