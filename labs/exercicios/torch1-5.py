import torch

print("======= q1 =======")
"""1. Criação e Manipulação de Tensores
● Crie um tensor 1D, 2D e 3D com valores aleatórios.
● Altere o tipo do tensor para float32, int64 e bool.
● Crie um tensor de zeros, uns e identidade (eye).
● Use reshape, view e unsqueeze para mudar a forma do tensor.
"""
a = torch.randn(5)  # 1D
b = torch.randn(3, 4)  # 2D
c = torch.randn(2, 3, 4)  # 3D
a_float = a.to(torch.float32)
a_bool = a.to(torch.bool)
b_int = b.to(torch.int64)
zeros = torch.zeros(3, 3)
ones = torch.ones(2, 2)
identity = torch.eye(4)
reshaped = b.view(3, -1)
unsqueezed = a.unsqueeze(0)  # acrescenta dimensão

print("======= q2 =======")
"""2. Indexação e Fatiamento
● Extraia uma linha, coluna e submatriz de um tensor 2D.
● Altere todos os valores maiores que 0.5 para 1, e o resto para 0.
● Use masked_select com uma máscara booleana.
"""
matrix = torch.randn(4, 4)
row = matrix[1]
col = matrix[:, 2]
sub = matrix[1:3, 1:3]
binary = (matrix > 0.5).float()
masked = torch.masked_select(matrix, matrix > 0)

print("======= q3 =======")
"""3. Operações Matemáticas
● Some dois tensores de mesma forma.
● Faça produto escalar e produto matricial.
● Calcule a média, soma, mínimo e máximo de um tensor.
● Normalize um tensor (subtraia média e divida pelo desvio padrão).
"""
x = torch.tensor([1., 2., 3.])
y = torch.tensor([4., 5., 6.])
sum_ = x + y
dot = torch.dot(x, y)
mat1 = torch.randn(2, 3)
mat2 = torch.randn(3, 4)
matmul = torch.matmul(mat1, mat2)
mean = x.mean()
std = x.std()
normalized = (x - mean) / std

print("======= q4 =======")
"""4. Operações Lógicas e Comparações
● Compare dois tensores com ==, >, < e obtenha a soma de elementos verdadeiros.
● Use torch.where para substituir valores com base em uma condição.
"""
a = torch.tensor([1, 2, 3])
b = torch.tensor([2, 2, 2])
equal = (a == b)
greater = (a > b)
true_count = (a > 1).sum()
cond = a > 2
replaced = torch.where(cond, a, torch.tensor(0))

print("======= q5 =======")
"""5. Broadcasting
● Some um vetor a cada linha de uma matriz.
● Multiplique um vetor coluna por uma matriz (broadcast automático).
"""
mat = torch.randn(3, 4)
vec = torch.tensor([1.0, 2.0, 3.0, 4.0])
added = mat + vec  # broadcasted addition

col_vec = torch.tensor([[1.0], [2.0], [3.0]])
result = col_vec * vec  # outer product shape (3, 4)
