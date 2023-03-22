from jax import jit, lax, grad

import jax.numpy as jnp

from tqdm import tqdm

import matplotlib.pyplot as plt

import optax


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

    initial = jnp.ones(len(fval), dtype=CDTYPE) / jnp.sqrt(dim)

    def layer(carry, p):
        y = fwht(carry * jnp.exp(1j * fval * p[1]))
        r = fwht(y * jnp.exp(1j * zrot * p[0]))
        return r, None

    def qaoa(params):
        res, _ = lax.scan(layer, initial, params)
        return res

    return qaoa

def norm(r):
    return jnp.sqrt(jnp.sum(jnp.abs(r) ** 2))

def get_masked_qaoa(fval: jnp.ndarray, mask: jnp.ndarray):
    dim = len(fval)
    N = int(jnp.log2(len(fval)))
    zrot = get_zrot(N)

    initial = jnp.ones(len(fval), dtype=CDTYPE) / jnp.sqrt(dim)
    initial *= mask

    initial /= jnp.sqrt((jnp.abs(initial) ** 2).sum())

    def layer(carry, p):
        y = fwht(carry * jnp.exp(1j * fval * p[1]))
        rb = jnp.exp(1j * zrot * p[0])
        r = fwht(y * rb)
        r2 = (1 - mask) * r
        r = mask * r
        r /= norm(r)
        return r, norm(r2) ** 2

    def qaoa(params):
        return lax.scan(layer, initial, params)

    return qaoa

def expval(fval: jnp.ndarray, qaoa):
    def f(params):
        x, _ = qaoa(params)
        return (jnp.abs(x) ** 2).dot(fval)
    return f

def get_inital(depth, T=1.0):
    dt = T / depth
    r = (jnp.arange(depth) + 1) / depth
    gammas = r * dt
    betas = (1 - r + 1 / depth) * dt
    return jnp.stack((betas, gammas)).T

x = jnp.linspace(-3, 3, 2**10)
fval = -jnp.exp(-x ** 2)
alpha = 0.0
mask = (jnp.abs(x - 0.5) > 0.8) * (1-alpha) + alpha

params = get_inital(10000, 10)
print(params)

qaoa = get_masked_qaoa(fval, mask)
# qaoa = get_qaoa(fval)

eqaoa = expval(fval, qaoa)

jqaoa = jit(eqaoa)
jqaoa(params).block_until_ready()
# gqaoa = jit(grad(eqaoa))
# print(jqaoa(params))
#
# opt = optax.adam(1e-2)
# opt_state = opt.init(params)
# p1 = []
# for _ in tqdm(range(200)):
#     g = gqaoa(params)
#     updates, opt_state = opt.update(g, opt_state, params)
#     params = optax.apply_updates(params, updates)
#     p1.append(jqaoa(params))

print(jqaoa(params))

# plt.plot(p1)
# plt.show()

plt.plot(fval)
plt.plot(-mask, color="tab:green")

res, b = jit(qaoa)(params)
ax = plt.twinx()
ax.plot(jnp.abs(res) ** 2, color="tab:orange")
plt.show()

print(jnp.prod(1-b))

