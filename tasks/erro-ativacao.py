import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)

# ============================================================================
# 1) FUNÇÕES DE ATIVAÇÃO E SUAS DERIVADAS
# ============================================================================
# O que realmente passa pela rede durante o retropropagação é a DERIVADA da
# ativação. Se ela for zero (degrau) ou muito pequena nas pontas (sigmoide e
# tanh saturadas), o gradiente "desaparece" e a camada anterior quase não
# aprende. Por isso o degrau (usado no Perceptron clássico) não serve para
# treinar uma MLP por retropropagação.

def linear(z):
    return z

def linear_deriv(z):
    return torch.ones_like(z)

def degrau(z):
    return (z >= 0).float()

def degrau_deriv(z):
    return torch.zeros_like(z)  # gradiente nulo quase sempre

def sigmoide(z):
    return torch.sigmoid(z)

def sigmoide_deriv(z):
    s = sigmoide(z)
    return s * (1 - s)

def tanh(z):
    return torch.tanh(z)

def tanh_deriv(z):
    return 1 - torch.tanh(z) ** 2

def relu(z):
    return torch.relu(z)

def relu_deriv(z):
    return (z > 0).float()

ATIVACOES = {
    "Linear": (linear, linear_deriv),
    "Degrau": (degrau, degrau_deriv),
    "Sigmoide": (sigmoide, sigmoide_deriv),
    "Tanh": (tanh, tanh_deriv),
    "ReLU": (relu, relu_deriv),
}

def plot_ativacoes():
    z = torch.linspace(-5, 5, 500)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for nome, (f, _) in ATIVACOES.items():
        ax1.plot(z, f(z), label=nome)
    ax1.set_title("Funções de ativação")
    ax1.legend()
    ax1.grid(True)

    for nome, (_, fd) in ATIVACOES.items():
        ax2.plot(z, fd(z), label=nome)
    ax2.set_title("Derivadas (o que a retropropagação usa de fato)")
    ax2.legend()
    ax2.grid(True)
    fig.tight_layout()

# ============================================================================
# 2) GRADIENTE DESCENDENTE: MÍNIMOS LOCAIS/GLOBAL E TAXA DE APRENDIZADO
# ============================================================================
# f(x) tem um vale raso (mínimo local) e um vale fundo (mínimo global). Para
# onde o gradiente descendente converge depende de ONDE se começa e do
# TAMANHO do passo (learning rate): pequeno demais é lento, grande demais
# oscila ou diverge.

def f_nao_convexa(x):
    return 0.1 * x ** 4 - x ** 2 + 0.2 * x

def gradiente_descendente(x0, lr, passos=60):
    x = torch.tensor(float(x0), requires_grad=True)
    caminho = [x.item()]
    for _ in range(passos):
        y = f_nao_convexa(x)
        y.backward()
        with torch.no_grad():
            x -= lr * x.grad
        x.grad.zero_()
        caminho.append(x.item())
        if not torch.isfinite(x):
            break  # divergiu: para antes do backward() quebrar com inf/nan
    return caminho

def plot_gradiente_descendente():
    x = torch.linspace(-3, 3, 400)
    y = f_nao_convexa(x)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(x, y, color="black")
    for x0 in [-2.5, -0.5, 0.5, 2.5]:
        caminho = torch.tensor(gradiente_descendente(x0, lr=0.05))
        ax1.plot(caminho, f_nao_convexa(caminho), marker=".", markersize=4,
                 label=f"início={x0}")
        print(f"[mínimos locais] início={x0:>5}: parou em x={caminho[-1]:.3f}")
    ax1.set_title("Mesmo LR, início diferente\n-> mínimo local x global")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(x, y, color="black")
    for lr in [0.01, 0.1, 1.0]:
        caminho_completo = gradiente_descendente(x0=2.5, lr=lr, passos=40)
        divergiu = len(caminho_completo) <= 40  # parou antes do fim = ficou infinito
        caminho = torch.tensor(caminho_completo[:-1] if divergiu else caminho_completo)
        ax2.plot(caminho, f_nao_convexa(caminho), marker=".", markersize=4,
                 label=f"lr={lr}" + (" (diverge)" if divergiu else ""))
        print(f"[learning rate] lr={lr:<5}: "
              + (f"DIVERGIU (|x|→∞) após {len(caminho)} passos" if divergiu
                 else f"parou em x={caminho[-1]:.3f}"))
    ax2.set_ylim(f_nao_convexa(x).min() - 1, 15)
    ax2.set_title("Mesmo início, LR diferente\n-> devagar / bom / diverge")
    ax2.legend()
    ax2.grid(True)
    fig.tight_layout()

# ============================================================================
# 3) SUPERFÍCIE DE ERRO CONVEXA (REGRESSÃO LINEAR)
# ============================================================================
# Ao contrário da função acima, o erro (MSE) de uma regressão linear em
# função de (w, b) é uma tigela convexa com um único mínimo global: o
# gradiente descendente sempre chega lá, não importa onde comece.

def gerar_dados_regressao(n=60, w_real=2.0, b_real=1.0, ruido=0.3):
    x = torch.linspace(-3, 3, n)
    y = w_real * x + b_real + ruido * torch.randn(n)
    return x, y

def plot_superficie_erro():
    x, y = gerar_dados_regressao()

    ws = torch.linspace(-1, 5, 120)
    bs = torch.linspace(-3, 5, 120)
    W, B = torch.meshgrid(ws, bs, indexing="ij")
    pred = W.unsqueeze(-1) * x + B.unsqueeze(-1)  # (len(ws), len(bs), n)
    E = ((pred - y) ** 2).mean(dim=-1)

    w = torch.tensor(-0.5, requires_grad=True)
    b = torch.tensor(-2.0, requires_grad=True)
    caminho_w, caminho_b = [w.item()], [b.item()]
    for _ in range(80):
        erro = ((w * x + b - y) ** 2).mean()
        erro.backward()
        with torch.no_grad():
            w -= 0.05 * w.grad
            b -= 0.05 * b.grad
        w.grad.zero_()
        b.grad.zero_()
        caminho_w.append(w.item())
        caminho_b.append(b.item())

    plt.figure(figsize=(6, 5))
    plt.contourf(W, B, E, levels=40, cmap="viridis")
    plt.colorbar(label="MSE")
    plt.plot(caminho_w, caminho_b, color="red", marker=".", markersize=4,
             label="caminho do gradiente descendente")
    plt.scatter([2.0], [1.0], color="white", marker="*", s=150,
                label="ótimo real (w=2, b=1)")
    plt.xlabel("w")
    plt.ylabel("b")
    plt.title("Superfície de erro convexa (regressão linear)")
    plt.legend()

# ============================================================================
# 4) PERCEPTRON MULTICAMADA (MLP), ATIVAÇÕES E RETROPROPAGAÇÃO
# ============================================================================
# XOR não é linearmente separável: nenhuma reta separa as duas classes. Duas
# condições SEPARADAS são necessárias para resolvê-lo com uma MLP:
#   (a) a ativação da camada oculta precisa ser não-linear (senão a
#       composição de camadas continua sendo uma transformação linear, não
#       importa quantos neurônios existam);
#   (b) é preciso mais de 1 neurônio oculto (1 neurônio só consegue traçar
#       UMA fronteira/reta; XOR precisa de pelo menos 2 para "recortar" as
#       duas regiões positivas).
# Os dois experimentos abaixo variam uma condição de cada vez, mantendo a
# outra fixa, para mostrar o efeito de cada uma isoladamente.

def gerar_dados_xor(n_por_classe=50, ruido=0.15):
    centros = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    rotulos_centro = torch.tensor([0., 1., 1., 0.])  # XOR
    xs, ys = [], []
    for centro, rotulo in zip(centros, rotulos_centro):
        xs.append(centro + ruido * torch.randn(n_por_classe, 2))
        ys.append(rotulo.repeat(n_por_classe))
    return torch.cat(xs), torch.cat(ys).view(-1, 1)

class MLP(torch.nn.Module):
    """Uma camada oculta com ativação e nº de neurônios configuráveis + saída sigmoide."""

    def __init__(self, ativacao=torch.tanh, n_ocultos=8):
        super().__init__()
        self.oculta = torch.nn.Linear(2, n_ocultos)
        self.saida = torch.nn.Linear(n_ocultos, 1)
        self.ativacao = ativacao

    def forward(self, x):
        h = self.ativacao(self.oculta(x))
        return torch.sigmoid(self.saida(h))

def treinar(model, x, y, epocas, lr=0.5):
    otimizador = torch.optim.SGD(model.parameters(), lr=lr)
    criterio = torch.nn.BCELoss()
    errors = []
    for _ in range(epocas):
        otimizador.zero_grad()
        pred = model(x)
        erro = criterio(pred, y)
        erro.backward()  # retropropagação: calcula d(erro)/d(peso)
        otimizador.step()  # gradiente descendente: peso -= lr * gradiente
        errors.append(erro.item())
    return errors

def plot_fronteira(ax, model, x, y, titulo):
    xx, yy = torch.meshgrid(torch.linspace(-0.5, 1.5, 200), torch.linspace(-0.5, 1.5, 200), indexing="ij")
    grade = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    with torch.no_grad():
        z = model(grade).view(xx.shape)
    ax.contourf(xx, yy, z, levels=[0, 0.5, 1], colors=["#f4a3a3", "#a3c9f4"])
    ax.scatter(x[:, 0], x[:, 1], c=y.view(-1), cmap="coolwarm", edgecolors="k")
    ax.set_title(titulo)

def plot_mlp_ativacoes(x, y):
    # (a) 8 neurônios ocultos fixos (modelo com bastante capacidade); variando apenas a função de ativação.
    # Linear nunca separa o XOR, não importa quantas épocas - continua sendo uma reta.
    # As não-lineares resolvem, mas em número de épocas diferente.
    torch.manual_seed(0)
    epocas = 2000
    ativacoes_mlp = {"Linear": linear, "Sigmoide": sigmoide, "Tanh": tanh, "ReLU": relu}
    modelos, errors = {}, {}
    for nome, ativ in ativacoes_mlp.items():
        torch.manual_seed(0)
        m = MLP(ativacao=ativ, n_ocultos=8)
        errors[nome] = treinar(m, x, y, epocas=epocas)
        modelos[nome] = m

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for ax, nome in zip(axs, ativacoes_mlp):
        plot_fronteira(ax, modelos[nome], x, y, nome)
    fig.suptitle("(a) 8 neurônios ocultos, variando a ativação")
    fig.tight_layout()

    plt.figure(figsize=(6, 4))
    for nome, p in errors.items():
        plt.plot(p, label=nome)
    plt.xlabel("época")
    plt.ylabel("erro (BCE)")
    plt.title("(a) A função de ativação escolhida pode acelerar o aprendizado\n(Linear (sem função) nunca resolve)")
    plt.legend()
    plt.grid(True)

    print("\n[ativação, 8 neurônios] erro final:")
    for nome, p in errors.items():
        print(f"  {nome:<9}: {p[-1]:.4f}")

def plot_mlp_n_neuronios(x, y):
    # (b) ativação não-linear fixa (Tanh); varia só o nº de neurônios ocultos.
    # Com 1 neurônio, a rede só consegue traçar UMA fronteira de decisão e
    # portanto nunca separa o XOR (que precisa de 2 fronteiras), mesmo sendo
    # não-linear e mesmo com muitas épocas.
    epocas = 1500
    tamanhos = [1, 3, 8]
    modelos, errors = {}, {}
    for n in tamanhos:
        torch.manual_seed(0)
        m = MLP(ativacao=tanh, n_ocultos=n)
        errors[n] = treinar(m, x, y, epocas=epocas)
        modelos[n] = m

    fig, axs = plt.subplots(1, len(tamanhos), figsize=(12, 4))
    for ax, n in zip(axs, tamanhos):
        plot_fronteira(ax, modelos[n], x, y, f"{n} neurônio(s) oculto(s)")
    fig.suptitle("(b) Função de ativação Tanh fixa, variando nº de neurônios ocultos")
    fig.tight_layout()

    plt.figure(figsize=(6, 4))
    for n, p in errors.items():
        plt.plot(p, label=f"{n} neurônio(s)")
    plt.xlabel("época")
    plt.ylabel("erro (BCE)")
    plt.title("(b) 1 neurônio nunca resolve o XOR\n(não-linearidade sozinha não basta)")
    plt.legend()
    plt.grid(True)

    print("\n[nº de neurônios, função de ativação Tanh] erro final:")
    for n, p in errors.items():
        print(f"  {n} neurônio(s): {p[-1]:.4f}")

def plot_mlp_xor():
    x, y = gerar_dados_xor()
    plot_mlp_ativacoes(x, y)
    plot_mlp_n_neuronios(x, y)

print("=== 1) Funções de ativação e derivadas ===")
plot_ativacoes()

print("\n=== 2) Gradiente descendente: mínimos locais e learning rate ===")
# plot_gradiente_descendente()

print("\n=== 3) Superfície de erro convexa (regressão linear) ===")
# plot_superficie_erro()

print("\n=== 4) MLP, ativações e retropropagação (XOR) ===")
# plot_mlp_xor()

plt.show()
