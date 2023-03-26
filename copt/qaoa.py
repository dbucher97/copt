from functools import partial
import jax.numpy as jnp
import jax
import numpy as np

from copt.objects import Problem
# from copt.fwht import fwht

import optax

CDTYPE = jnp.complex64
DTYPE = jnp.float32


def get_inital(depth, T=1.0):
    dt = T / depth
    r = (jnp.arange(depth) + 1) / depth
    gammas = r * dt
    betas = (1 - r + 1 / depth) * dt
    return jnp.stack((betas, gammas)).T

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

def build_mixer(n):
    a = jnp.array([-0.5, 0.5], dtype=DTYPE)
    ret = jnp.array([0], dtype=DTYPE)
    for _ in range(n):
        ret = ret[:, None] + a[None, :]
        ret = ret.ravel()
    return ret


def get_layer(objective, mixer):
    @jax.jit
    def layer(x, params):
        x = x * jnp.exp(-1j * objective * params[1])
        x = fwht(x)
        x = x * jnp.exp(-1j * mixer * params[0])
        x = fwht(x)
        return x

    return layer


def get_qaoa(objective, mixer):
    initial = jnp.ones_like(objective, dtype=CDTYPE) / jnp.sqrt(objective.shape[0])
    layer = get_layer(objective, mixer)

    @jax.jit
    def qaoa(params):
        x, y = jax.lax.scan(lambda c, p: (layer(c, p), 1.0), initial, params)
        return x, jnp.prod(y)

    return qaoa


def get_masked_qaoa(objective, mixer, mask):
    initial = mask * jnp.ones_like(objective, dtype=CDTYPE)
    initial = initial / jnp.sqrt((jnp.abs(initial) ** 2).sum())
    vanilla_layer = get_layer(objective, mixer)

    m = jnp.sqrt(mask)

    @jax.jit
    def layer(c, p):
        x = vanilla_layer(c, p)
        y = m * x
        prob = (jnp.abs(y) ** 2).sum()
        return y / jnp.sqrt(prob), prob

    @jax.jit
    def qaoa(params):
        x, y = jax.lax.scan(lambda c, p: layer(c, p), initial, params)
        return x, jnp.prod(y)

    return qaoa


def get_loss(qaoa, objective):
    @jax.jit
    def loss(params):
        x, p = qaoa(params)
        x = jnp.abs(x) ** 2
        return x.dot(objective), p

    return loss


def train(loss, initial, lr: float = 1e-2, steps: int = 1000):
    params = initial
    trace = []
    lf = jax.jit(jax.value_and_grad(loss, has_aux=True))
    opt = optax.adam(lr)
    opt_state = opt.init(params)
    from tqdm import tqdm
    for _ in tqdm(range(steps)):
        v, g = lf(params)
        updates, opt_state = opt.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)
        trace.append(v)
    return params, jnp.array(trace)


def to_penalty_qaoa(
    problem: Problem, depth: int = 1, extra_factor: float = 1.0, T: float = 1.0
):
    objective = problem.obj_values
    for c in problem.constraints:
        # a = jnp.min(jnp.where(c.feasible, problem.obj_values, jnp.inf))
        # xi = (a - problem.obj_values) / c.penalty_tesselation
        # mv = jnp.max(jnp.where(c.feasible, -jnp.inf, xi))
        objective = objective + extra_factor * c.penalty_tesselation
    objective = objective / jnp.max(objective)

    mixer = build_mixer(problem.domain.num_qubits)
    initial = get_inital(depth, T=T)
    circ = get_qaoa(objective, mixer)

    return circ, initial, get_loss(circ, objective)


def to_masked_qaoa(problem: Problem, depth: int = 1, T: float = 1.0, exact=False):
    objective = problem.obj_values
    objective = objective / jnp.max(objective)

    mask = jnp.ones_like(objective)
    for c in problem.constraints:
        if exact:
            mask = mask * c.feasible
        else:
            mask = mask * c.constrained_measure

    mixer = build_mixer(problem.domain.num_qubits)
    initial = get_inital(depth, T=T)
    circ = get_masked_qaoa(objective, mixer, mask)

    return circ, initial, get_loss(circ, objective)
