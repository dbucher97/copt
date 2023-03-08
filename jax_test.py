from jax import jit

import jax.numpy as jnp

DTYPE = jnp.complex64

@jit
def fwht(x):
    d = x.shape[0]
    h = 2
    while h <= d:
        hf = h // 2
        x = x.reshape((-1, h))
        half_1, half_2 = x[:, :hf], x[:, hf:]
        x = jnp.hstack((half_1 + half_2, half_1 - half_2))
        h *= 2

    return (x / jnp.sqrt(d)).reshape((d,))

def get_zrot(N: int) -> jnp.ndarray:
    zrot = jnp.array([0.], dtype=DTYPE)
    rz = jnp.array([-0.5, 0.5], dtype=DTYPE)
    for _ in range(N):
        zrot = zrot[:, None] + rz[None, :]
        zrot = zrot.ravel()
    return zrot


def get_qaoa_layer(fval: jnp.ndarray):
    N = int(jnp.log2(len(fval)))
    zrot = get_zrot(N)

    def qaoa_layer(x, params):
        jnp.exp(fwht(x * jnp.exp(1j * fval * params[0])) * para,
        x = fwht(x)
        x *= 
        return fwht(x)

    return jit(qaoa_layer)

def get_qaoa(fval: jnp.ndarray):
    layer = get_qaoa_layer(fval)
    def qaoa(params):
        x = jnp.ones(len(fval), dtype=DTYPE) / jnp.sqrt(len(fval))
        for p in params:
            x = layer(x, p)
        return jnp.abs(x).dot(fval)
    return jit(qaoa)

fval = jnp.exp(-jnp.linspace(-1, 1, 2 ** 10) ** 2)

params = jnp.ones((30, 2))

qaoa = get_qaoa(fval)

