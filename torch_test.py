import torch
import numpy as np

DTYPE = torch.complex64

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


def get_qaoa_layer(fval: torch.Tensor):
    N = int(np.log2(len(fval)))
    zrot = get_zrot(N)

    def qaoa_layer(x, params):
        x *= torch.exp(1j * fval * params[0])
        x = fwht(x)
        x *= torch.exp(1j * zrot * params[1])
        return fwht(x)

    return qaoa_layer

def get_qaoa(fval: torch.Tensor):
    layer = get_qaoa_layer(fval)
    def qaoa(params):
        x = torch.ones(len(fval), dtype=DTYPE) / np.sqrt(len(fval))
        for p in params:
            x = layer(x, p)
        return torch.abs(x).dot(fval)
    return qaoa

fval = torch.tensor(np.exp(-np.linspace(-1, 1, 2 ** 10) ** 2), dtype=torch.float32)

params = torch.ones((30, 2))

qaoa = get_qaoa(fval)
