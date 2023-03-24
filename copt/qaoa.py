import jax.numpy as jnp
import jax
import numpy as np

from copt.objects import Problem

CDTYPE = jnp.complex64
DTYPE = jnp.float32

def fwht(x):
    d = x.shape[0]
    maxit = int(np.ceil(np.log2(d)))
    print(maxit)

    def body(i, x):
        hf = 2 ** i
        h = 2 * hf
        print(h)
        a = x[0::h]
        b = x[hf::h]
        a = a + b
        b = a - b
        # x = jax.lax.dynamic_update_slice_in_dim(x, a, 0, d, h)
        # x = jax.lax.dynamic_update_slice_in_dim(x, b, hf, d, h)
        print(a.shape)
        # half_1, half_2 = x[:, :hf], x[:, hf:]
        # x = jnp.hstack((half_1 + half_2, half_1 - half_2))
        return x * (i + 1) / h

    body(0, x)
    x = jax.lax.fori_loop(0, maxit, body, x)
    print(x)

    return (x / jnp.sqrt(d)).reshape((d,))

def to_penalty_qaoa(problem: Problem, extra_factor: float = 1.):
    objective = problem.obj_values
    for c in problem.constraints:
        a = jnp.min(jnp.where(c.feasible, problem.obj_values, jnp.inf))
        xi = (a - problem.obj_values) / c.penalty_tesselation
        mv = jnp.max(jnp.where(c.feasible, -jnp.inf, xi))
        objective = objective + mv * extra_factor * c.penalty_tesselation

    i = jnp.array(np.random.rand(2 ** 10))

    fwht(i)
    print(jax.jit(fwht)(i))

