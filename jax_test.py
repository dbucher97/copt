from jax import jit, lax, grad

import jax.numpy as jnp


CDTYPE = jnp.complex64
DTYPE = jnp.float32

def fwht(x):
    d = x.shape[0]

    h = 2

    while h <= d:
        hf = h // 2
        x = x.reshape((-1, h))
        half_1, half_2 = x[:, :hf], x[:, hf:]
        x = jnp.hstack((half_1 + half_2, half_1 - half_2))
        h = 2 * h

    return (x / jnp.sqrt(d)).reshape((d,))


def get_zrot(N: int):
    zrot = jnp.array([0.0], dtype=DTYPE)
    rz = jnp.array([-0.5, 0.5], dtype=DTYPE)
    for _ in range(N):
        zrot = zrot[:, None] + rz[None, :]
        zrot = zrot.ravel()
    return zrot


def get_qaoa(fval: jnp.ndarray):
    dim = len(fval)
    N = int(jnp.log2(len(fval)))
    zrot = get_zrot(N)

    def qaoa(params):
        initial = jnp.ones(len(fval), dtype=CDTYPE) / jnp.sqrt(dim)

        def layer(carry, p):
            y = fwht(carry * jnp.exp(1j * fval * p[1]))
            r = fwht(y * jnp.exp(1j * zrot * p[0]))
            return r, None

        res, _ = lax.scan(layer, initial, params)
        return jnp.abs(res).dot(fval)

    return qaoa


fval = jnp.exp(-jnp.linspace(-1, 1, 2**21) ** 2)

params = jnp.ones((10, 2))

qaoa = get_qaoa(fval)

jqaoa = jit(qaoa)
jqaoa(params).block_until_ready()
print("compiled")

import timeit
num=1
r = timeit.timeit(lambda: jqaoa(params).block_until_ready(), number=num)
print(r / num)

gqaoa = jit(grad(qaoa))
print("grad")
gqaoa(params)
print("compiled")
num=1
r = timeit.timeit(lambda: gqaoa(params).block_until_ready(), number=num)
print(r / num)
