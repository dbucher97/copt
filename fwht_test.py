from functools import partial
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsp
import scipy.sparse as sp
import numpy as np
import time

h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


def get_h_at(N, i):
    if i == 0:
        return sp.kron(h, sp.eye(2 ** (N - 1)))
    elif i == N - 1:
        return sp.kron(sp.eye(2 ** (N - 1)), h)
    else:
        return sp.kron(sp.kron(sp.eye(2**i), h), sp.eye(2 ** (N - i - 1)))


@jax.jit
def fwht(x, op):
    d = x.shape[0]

    h = 2

    while h <= d:
        hf = h // 2
        x = x.reshape((-1, h))
        half_1, half_2 = x[:, :hf], x[:, hf:]
        x = jnp.hstack(
            (
                half_1 * op[0, 0] + half_2 * op[0, 1],
                half_1 * op[0, 1] + half_2 * op[1, 1],
            )
        )
        h = 2 * h

    return x.reshape((d,))


N = 20
# spmats = [jsp.BCOO.from_scipy_sparse(get_h_at(N, i)) for i in range(N)]


@jax.jit
def sph(x):
    d = x.shape[0]
    h = 2
    i = 0
    while h <= d:
        x = spmats[i] @ x
        h *= 2
        i += 1
    return x


x = jnp.ones(2**N) / jnp.sqrt(2**N)
origshape = x.shape


n = 100


def timed(f, *args):
    a = time.time()
    for _ in range(n):
        f(*args).block_until_ready()
    return (time.time() - a) / n


# print(timed(sph, x))
# print(timed(fwht, x))
# print(timed(jax.grad(lambda x: sph(x).sum()), x))
# print(timed(jax.grad(lambda x: fwht(x).sum()), x))

h = jnp.array(h)


@jax.jit
def apply_to_all(x, op):
    for i in range(0, N):
        shape = [2] * N
        x = x.reshape((2, -1))
        x = op @ x
        x = x.reshape(shape)
        if i < N - 1:
            x = x.swapaxes(0, i + 1)
        else:
            x = x.transpose((i % N for i in range(N - 1, 2 * N - 1)))
    return x.flatten()


# print(timed(fwht, x, h))
# print(timed(apply_to_all, x, h))
# print(timed(jax.grad(lambda x, y: fwht(x, y).sum()), x, h))
# print(timed(jax.grad(lambda x, y: apply_to_all(x, y).sum()), x, h))

a = time.time()

@jax.jit
def fft(x):
    return jnp.fft.fft(x)

for _ in range(n):
    fft(x).block_until_ready()
print ((time.time() - a) / n)
