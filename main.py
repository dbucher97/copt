import time

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

from copt.objects import LinSpace, Constraint, Problem, Domain
from copt.qaoa import to_penalty_qaoa, to_masked_qaoa, train

import numpy as np
import jax.numpy as jnp


def plot_inequality():
    a = LinSpace(space=(np.pi, -np.pi), num_qubits=10, endpoint=False)
    domain = Domain([a])
    sns.set(palette="deep", style="ticks", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(10, 5))
    for q in [8, 5, 3]:
        c = Constraint(
            domain,
            lhs=lambda x: x,
            rhs=0,
            num_qubits=q,
            underdrive=1,
            pint=a.num_qubits - q,
        )
        prob = c.constrained_measure.reshape(domain.shape)
        ax.plot(a.array, prob, label=f"$M={q}$", linewidth=3)
    ax.set_xlabel(r"$\varphi$")
    ax.set_ylabel(r"Probability $P(\varphi \leq 0)$")
    ax.legend()
    plt.savefig("plots/ineq_prob.pdf", bbox_inches="tight", transparent=True)


def plot_example1():
    a = LinSpace(space=(-1, 1), num_qubits=4)

    domain = Domain([a, a])
    sns.set(palette="deep", style="ticks", font_scale=1.5)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    c = Constraint(
        domain,
        lhs=lambda x: x[0] + x[1],
        rhs=1,
        num_qubits=8,
    )

    def fun(i):
        x, y = i
        g1 = jnp.exp(- 2 * ((x + 0.8) ** 2 + (y + 0.8) ** 2 )) / 1.5
        g2 = jnp.exp(- 0.5 * (x - 0.5) ** 2 - (y - 0.5) ** 2)
        return -g1 -g2

    p = Problem(
        domain,
        fun,
        constraints=[c],
    )

    ax[0].matshow(p.obj_values.reshape(domain.shape), cmap="bwr")
    print(c.feasible)
    ax[1].matshow(jnp.where(c.feasible, p.obj_values, jnp.nan).reshape(domain.shape) , cmap="bwr")

    plt.savefig("plots/example1.pdf", bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    # plot_inequality()
    plot_example1()
    # domain = Domain([a, a])
    #
    # c = Constraint(domain, lhs=lambda x: x[0] + x[1], rhs=0, num_qubits=8)
    #
    # p = Problem(
    #     domain,
    #     lambda x: 0.5 * (x[0] - 0.4) ** 2 + (x[1] - 0.3) ** 2,
    #     constraints=[c],
    # )
    #
    # x = to_penalty_qaoa(p, 10)
    #
    # plt.matshow(x.reshape(domain.shape), norm="log")
    # plt.show()

#     v, pos = p.get_min()
#     print(pos)
#
#
#     x = jnp.where(p.feasible, p.obj_values, jnp.nan)
#
#     plt.matshow(x.reshape(domain.shape), cmap="bwr")
#     rect = Rectangle(
#         jnp.array(pos)[::-1] - 0.51,
#         1,
#         1,
#         linewidth=2,
#         edgecolor="r",
#         facecolor="none",
#     )
#     plt.gca().add_patch(rect)
#     plt.show()
#
#     depth = 100
#     # q, i, l = to_penalty_qaoa(p, depth=depth, T=1, extra_factor=10)
#     q, i, l = to_masked_qaoa(p, depth=depth, exact=False, T=10)
#
#     i, t = train(l, i, steps=500)
#
#     plt.plot(t)
#     plt.show()
#
#     x, p = q(i)
#     x = jnp.abs(x) ** 2 * p
#
#     plt.matshow(x.reshape(domain.shape), cmap="Blues")
#     rect = Rectangle(
#         jnp.array(pos)[::-1] - 0.51,
#         1,
#         1,
#         linewidth=2,
#         edgecolor="r",
#         facecolor="none",
#     )
#     plt.gca().add_patch(rect)
#     plt.show()
