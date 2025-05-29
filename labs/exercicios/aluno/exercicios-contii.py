import torch

print("\n============= Tensor 1D =============")
tensor_1d = torch.rand(5)
print(tensor_1d, "\nDimensões:", tensor_1d.ndim)
print("Tipo:", tensor_1d.dtype)
tensor_float32 = tensor_1d.to(torch.float32)
print("Transformacão float32:", tensor_float32.dtype)
tensor_int64 = tensor_1d.to(torch.int64)
print("Transformacão int64:", tensor_int64.dtype)
tensor_bool = tensor_1d.to(torch.bool)  # Valores maiores que zero viram True, zero vira False
print("Transformacão bool:", tensor_bool.dtype, "\nExemplo: ", tensor_bool)

print("\n============= Tensor 2D =============")
tensor_2d = torch.rand(3, 4)  # 3 linhas, 4 colunas
print(tensor_2d)
print("Dimensões:", tensor_2d.ndim)

print("\n============= Tensor 3D =============")
tensor_3d = torch.rand(2, 3, 4)  # 2 "cubos", de 3 linhas e 4 colunas
print(tensor_3d)
print("Dimensões:", tensor_3d.ndim)


print("\n======== Outras Manipulações ========")
zeros_tensor = torch.zeros(2, 5)
print("Tensor de zeros:\n", zeros_tensor)
ones_tensor = torch.ones(3)
print("\nTensor de uns:\n", ones_tensor)
identity_tensor = torch.eye(4) # Cria uma matriz identidade quadrada n×n, com uns na diagonal principal.
print("\nTensor identidade:\n", identity_tensor)
tensor_2d_reshaped = tensor_2d.reshape(4, 3)  # Transforma  o tensor 2d anterior (é possivel usar -1 em uma das dimensões).
print("\nTensor 2D reshape(4,3):\n", tensor_2d_reshaped)

# tensor_unsq = tensor_original.unsqueeze(0)  # Adiciona uma dimensão no início (transforma em um tensor 1x12)
# print("\nTensor unsqueeze(0):", tensor_unsq, "Dimensão:", tensor_unsq.ndim)
# tensor_unsq_mid = tensor_original.unsqueeze(1)  # Adiciona uma dimensão no meio (transforma em um tensor 12x1)
# print("Tensor unsqueeze(1):\n", tensor_unsq_mid, "Dimensão:", tensor_unsq_mid.ndim)

print("\n==== Linhas, colunas e submatriz ====")
print("Tensor 2D Reshape:\n", tensor_2d_reshaped.view(4, 3)) # visualização do tensor
print("\nTerceira Linha do Tensor 2D Reshape:\n", tensor_2d_reshaped[2,:])
print("\nTerceira Coluna do Tensor 2D Reshape:\n", tensor_2d_reshaped[:,2])
print("\nSubmatriz do Tensor 2D Reshape:\n", tensor_2d_reshaped[0:2, 0:2])

print("\n==== Limiarização (thresholding) ====")

print("\n === Usando manipulação Booleana:")
mascara = tensor_2d_reshaped > 0.5
print("Máscara Booleana (valores > 0.5):\n", mascara)
tensor_modificado = mascara.int() # Perceba a presença do dtype no output. Uma visualização importante ao trabalhar com tensores. 
print("Tensor Modificado:\n", tensor_modificado)

print("\n === Usando operações de comparação:") # Para alguns casos, pode ser mais indicado. torch.where é uma operação eficiente implementada no backend em C++.
condicao = tensor_2d_reshaped > 0.5
valor_se_verdadeiro = torch.tensor(1.0)  # Importante usar um float para corresponder ao tipo do tensor original, ou você pode converter depois
valor_se_falso = torch.tensor(0.0)      # Mesmo aqui
tensor_modificado_where = torch.where(condicao, valor_se_verdadeiro, valor_se_falso)
print("Tensor Modificado (torch.where):\n", tensor_modificado_where)

print("\n === Usando masked_select:")
elementos_selecionados = torch.masked_select(tensor_2d_reshaped, mascara)
print("Elementos Selecionados (onde a máscara é True):\n", elementos_selecionados)

print("\n==== Soma de dois tensores de mesma forma ====")
tensor_a = torch.rand(3, 4)
tensor_b = torch.rand(3, 4)
tensor_soma = tensor_a + tensor_b
print("Tensor A:\n", tensor_a)
print("Tensor B:\n", tensor_b)
print("Soma de A e B:\n", tensor_soma)

print("\n========= Produto escalar =========")
tensor_vetor1 = torch.tensor([1, 2, 3])
tensor_vetor2 = torch.tensor([4, 5, 6])
produto_escalar = torch.dot(tensor_vetor1.float(), tensor_vetor2.float()) # O produto escalar é definido apenas para tensores 1D.
# Há como definir operações entre tensores de dimensões maiores que 1, mas o resultado geralmente não é um simples escalar (Tensor Contraction / Produto Interno Generalizado / Soma de Produtos Elemento a Elemento).
print("Tensor Vetor 1:", tensor_vetor1)
print("Tensor Vetor 2:", tensor_vetor2)
print("Produto Escalar:", produto_escalar)

print("\n========= Produto matricial =========")
tensor_matriz1 = torch.rand(10, 2)
tensor_matriz2 = torch.rand(2, 3)
produto_matricial = torch.matmul(tensor_matriz1, tensor_matriz2) # multiplicação de matrizes gerando um tensor 2x2
# O produto matricial é definido apenas para tensores 2D, onde o número de colunas da primeira matriz deve ser igual ao número de linhas da segunda matriz ((10,2) com (2,3) também funciona).
print("\nTensor Matriz 1:\n", tensor_matriz1)
print("Tensor Matriz 2:\n", tensor_matriz2)
print("Produto Matricial:\n", produto_matricial)


print("\n ==== Cálculo de média, soma, mínimo e máximo de um tensor ====")
tensor_estatisticas = torch.tensor([1.0, 5.0, 2.0, 8.0, 3.0])
media = torch.mean(tensor_estatisticas)
soma = torch.sum(tensor_estatisticas)
minimo = torch.min(tensor_estatisticas)
maximo = torch.max(tensor_estatisticas)
print("Tensor para Estatísticas:", tensor_estatisticas)
print("Média:", media)
print("Soma:", soma)
print("Mínimo:", minimo)
print("Máximo:", maximo)


print("\n ==== Normalização de um tensor ====")
tensor_normalizar = torch.tensor([2.0, 4.0, 6.0, 8.0], dtype=torch.float32)
media_normalizar = torch.mean(tensor_normalizar)
desvio_padrao = torch.std(tensor_normalizar)
tensor_normalizado = (tensor_normalizar - media_normalizar) / desvio_padrao
print("Tensor Original:", tensor_normalizar)
print("Média:", media_normalizar)
print("Desvio Padrão:", desvio_padrao)
print("Tensor Normalizado:", tensor_normalizado)

