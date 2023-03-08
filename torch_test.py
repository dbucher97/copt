import torch
import numpy as np

CDTYPE = torch.complex64
DTYPE = torch.float32

def fwht(x):
    d = x.shape[0]
    h = 2
    while h <= d:
        hf = h // 2
        x = x.reshape((-1, h))
        half_1, half_2 = x[:, :hf], x[:, hf:]
        x = torch.hstack((half_1 + half_2, half_1 - half_2))
        h *= 2

    return (x / np.sqrt(d)).reshape((d,))

def get_zrot(N: int) -> torch.tensor:
    zrot = torch.tensor([0.], dtype=DTYPE)
    rz = torch.tensor([-0.5, 0.5], dtype=DTYPE)
    for _ in range(N):
        zrot = zrot[:, None] + rz[None, :]
        zrot = zrot.ravel()
    return zrot

def get_qaoa(fval: torch.Tensor):
    dim = len(fval)
    N = int(np.log2(len(fval)))
    zrot = get_zrot(N)

    def qaoa(params):
        initial = torch.ones(len(fval), dtype=CDTYPE) / np.sqrt(dim)

        def layer(carry, p):
            y = fwht(carry * torch.exp(1j * fval * p[1]))
            r = fwht(y * torch.exp(1j * zrot * p[0]))
            return r

        x = initial
        for p in params:
            x = layer(x, p)
        return torch.abs(x).dot(fval)

    return qaoa

fval = torch.tensor(np.exp(-np.linspace(-1, 1, 2 ** 21) ** 2), dtype=DTYPE)

params = torch.ones((50, 2), requires_grad=True)

qaoa = get_qaoa(fval)

import timeit
num=1
r = timeit.timeit(lambda: qaoa(params), number=num)
print(r / num)

num=1
ps = [torch.ones((20, 2), requires_grad=True) for _ in range(num)]
rs = [qaoa(pi) for pi in ps]

import time
a = time.time()
for ri in rs:
    ri.backward()
print((time.time()-a) / num)

