import torch
from hadamard_transform import hadamard_transform
from tqdm import tqdm
import numpy as np

N = 21

initial = torch.ones(2**N, dtype=torch.complex128) / 2 ** (N / 2)


x_base = np.array([0.])
for _ in range(N):
    x_base = x_base[:, None] + np.array([-1/2, 1/2])[None, :]
    x_base = x_base.ravel()
x_base = torch.tensor(x_base, dtype=torch.complex128)


target =  -torch.tensor(np.exp(-np.linspace(-5, 5, 2**N) ** 2))


k = 50

beta = 1
gamma = 0

delta = 0.02

state = initial


for i in tqdm(range(k)):
    state *= torch.exp(1j * gamma * target)

    state = hadamard_transform(state.real) + 1j * hadamard_transform(state.imag)
    state *= torch.exp(1j * beta * x_base)
    state = hadamard_transform(state.real) + 1j * hadamard_transform(state.imag)

    beta -= delta
    gamma += delta

import matplotlib.pyplot as plt
plt.plot(torch.abs(state))

plt.show()
# torch.exp(1j * initial * )

# print(res)
