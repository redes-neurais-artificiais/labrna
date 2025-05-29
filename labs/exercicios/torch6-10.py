import numpy as np
import torch

print("======= q6 =======")
"""6. Conversão com NumPy
● Converta um tensor para um array NumPy e vice-versa.
● Verifique compartilhamento de memória.
"""
t = torch.tensor([1.0, 2.0])
a = t.numpy()
t_back = torch.from_numpy(a)

print("======= q7 =======")
"""7. Gradientes e Autograd
● Crie um tensor com requires_grad=True.
● Calcule uma função escalar (ex: y=x2+3x+1y = x^2 + 3x + 1y=x2+3x+1).
● Use backward() para calcular os gradientes.
● Acesse x.grad.
"""
x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 3*x + 1
y.backward()
grad = x.grad  # dy/dx = 2x + 3

print("======= q8 =======")
"""
"""

print("======= q9 =======")
"""
"""

print("======= q10 =======")
"""
"""
