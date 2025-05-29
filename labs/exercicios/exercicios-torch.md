# Exercícios de torch em Python

## 1. Criação e Manipulação de Tensores

* Crie um tensor 1D, 2D e 3D com valores aleatórios.
* Altere o tipo do tensor para `float32`, `int64` e `bool`.
* Crie um tensor de zeros, uns e identidade (`eye`).
* Use `reshape`, `view` e `unsqueeze` para mudar a forma do tensor.

## 2. Indexação e Fatiamento

* Extraia uma linha, coluna e submatriz de um tensor 2D.
* Altere todos os valores maiores que 0.5 para 1, e o resto para 0.
* Use `masked_select` com uma máscara booleana.

## 3. Operações Matemáticas

* Some dois tensores de mesma forma.
* Faça produto escalar e produto matricial.
* Calcule a média, soma, mínimo e máximo de um tensor.
* Normalize um tensor (subtraia média e divida pelo desvio padrão).

## 4. Operações Lógicas e Comparações

* Compare dois tensores com `==`, `>`, `<` e obtenha a soma de elementos verdadeiros.
* Use `torch.where` para substituir valores com base em uma condição.

## 5. Broadcasting

* Some um vetor a cada linha de uma matriz.
* Multiplique um vetor coluna por uma matriz (broadcast automático).

## 6. Conversão com NumPy

* Converta um tensor para um array NumPy e vice-versa.
* Verifique compartilhamento de memória.

## 7. Gradientes e Autograd

* Crie um tensor com `requires_grad=True`.
* Calcule uma função escalar (ex: $y=x^2 + 3x + 1$).
* Use `backward()` para calcular os gradientes.
* Acesse `x.grad`.