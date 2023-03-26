import time

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from copt.objects import LinSpace, Constraint, Problem, Domain
from copt.qaoa import to_penalty_qaoa, to_masked_qaoa, train

import jax.numpy as jnp
import jax

if __name__ == "__main__":
    a = LinSpace(space=(-1.0, 1.0), num_qubits=4)
    domain = Domain([a, a])

    c = Constraint(
        domain,
        lhs=lambda x: x[0] + x[1],
        rhs=1,
        num_qubits=8,
    )

    p = Problem(
        domain,
        lambda x: (x[0] + x[1]) * (x[0] - x[1]),
        constraints=[c],
    )

    v, pos = p.get_min()
    print(pos)


    x = jnp.where(p.feasible, p.obj_values, jnp.nan)

    plt.matshow(x.reshape(domain.shape), cmap="bwr")
    rect = Rectangle(
        jnp.array(pos)[::-1] - 0.51,
        1,
        1,
        linewidth=2,
        edgecolor="r",
        facecolor="none",
    )
    plt.gca().add_patch(rect)
    plt.show()

    depth = 100
    # q, i, l = to_penalty_qaoa(p, depth=depth, T=1, extra_factor=10)
    q, i, l = to_masked_qaoa(p, depth=depth, exact=False, T=10)

    i, t = train(l, i, steps=500)

    plt.plot(t)
    plt.show()

    x, p = q(i)
    x = jnp.abs(x) ** 2 * p

    plt.matshow(x.reshape(domain.shape), cmap="Blues")
    rect = Rectangle(
        jnp.array(pos)[::-1] - 0.51,
        1,
        1,
        linewidth=2,
        edgecolor="r",
        facecolor="none",
    )
    plt.gca().add_patch(rect)
    plt.show()
