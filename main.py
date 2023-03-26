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


def example1(n1=4, n2=4):
    a = LinSpace(space=(-1, 1), num_qubits=n1)

    domain = Domain([a, a])

    c = Constraint(
        domain,
        lhs=lambda x: x[0] + x[1],
        rhs=0,
        num_qubits=n2,
    )

    def fun(i):
        x, y = i
        g1 = jnp.exp(-2 * ((x + 0.8) ** 2 + (y + 0.8) ** 2)) / 1.3
        g2 = jnp.exp(-0.5 * (x - 0.5) ** 2 - (y - 0.5) ** 2)
        return 1 - g1 - g2

    p = Problem(
        domain,
        fun,
        constraints=[c],
    )
    return p, c, domain


def example2(n1=7, n2=8):
    a = LinSpace(space=(-1, 1), num_qubits=n1)

    domain = Domain([a, a])

    def cfun(inp):
        x, y = inp
        return jnp.cos(8 * jnp.sqrt(x ** 2 + x*y + 2 * y ** 2))

    c = Constraint(
        domain,
        lhs=cfun,
        rhs=0,
        num_qubits=n2,
    )

    np.random.seed(2135132)

    pos = np.random.rand(6, 2) * 2 - 1
    height = np.random.rand(pos.shape[0]) * 0.2 + 0.8
    height *= np.random.randint(0, 2, pos.shape[0]) * 2 - 1
    width = np.random.rand(pos.shape[0]) * 0.4 + 0.8

    def fun(x):
        ret = jnp.cos(10 * x[0]) + jnp.sin(7 * x[1] + 1.2)
        ret *= jnp.exp(-(x[0] + np.pi/20 - 0.05) ** 2)
        ret *= jnp.exp(-(x[1] + np.pi/7) ** 2)
        # for p, h, w in zip(pos, height, width):
        #     r = jnp.sum((x - p)**2)
        #     ret += h * jnp.exp(-jnp.sqrt(r) / w)
        return ret

    p = Problem(
        domain,
        fun,
        constraints=[c],
    )
    return p, c, domain


def plot_example(fun):
    p, c, domain = fun()
    sns.set(palette="deep", font_scale=1.5, style="white")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    v, pos = p.get_min()
    rect = Rectangle(
        jnp.array(pos)[::-1] - 0.51,
        1,
        1,
        linewidth=3,
        edgecolor="#D84141",
        facecolor="none",
    )

    ax[0].matshow(p.obj_values.reshape(domain.shape), cmap="bwr")
    ax[1].matshow(
        jnp.where(c.feasible, p.obj_values, jnp.nan).reshape(domain.shape), cmap="bwr"
    )
    ax[1].add_patch(rect)

    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].set_xticks([])
    ax[1].set_yticks([])

    plt.savefig(f"plots/{fun.__name__}.pdf", bbox_inches="tight", transparent=True)


def opt_example(fun, typ="masked", anc=4):
    p, c, domain = fun(n2=anc)
    sns.set(palette="deep", font_scale=1.5, style="ticks")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    v, pos = p.get_min()
    rect = Rectangle(
        jnp.array(pos)[::-1] - 0.51,
        1,
        1,
        linewidth=3,
        edgecolor="#D84141",
        facecolor="none",
    )

    depth = 10
    T = 1
    if typ == "masked":
        q, i, l = to_masked_qaoa(p, depth=depth, T=T)
    else:
        q, i, l = to_penalty_qaoa(p, depth=depth, extra_factor=1.5, T=T)
    i, t = train(l, i, steps=500)

    x, p = q(i)
    x = jnp.abs(x) ** 2 * p
    x = x.reshape(domain.shape)

    ax[0].matshow(x, cmap="Blues")
    ax[0].add_patch(rect)
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].plot(t[:, 0], color="tab:blue", linewidth=4)
    ax[1].set_ylabel("Objective")
    if typ == "masked":
        ax2 = ax[1].twinx()
        ax2.plot(t[:, 1], color="tab:orange", linewidth=4, linestyle="dashed")
        ax2.set_ylabel("Probability")
        ax2.set_xlabel("Iteration")

    non_feas = 100 * jnp.where(c.feasible.reshape(domain.shape), 0, x).sum()
    a = 100 * x[pos]

    fig.suptitle(f"Measuring minimum: {a:.2f}%  Non-feasible outcomes: {non_feas:.2f}%")

    plt.savefig(
        f"plots/opt_{fun.__name__}_{typ}_{anc}.pdf", bbox_inches="tight", transparent=True
    )
    plt.show()


def plot_prob_num_qubits_example1():
    sns.set(palette="deep", font_scale=1.5, style="ticks")
    plt.subplots(figsize=(5, 4))

    for d in [5, 10, 20]:
        res = []
        for i in range(3, 15):
            print(i)
            p, _, _ = example1(n2=i)
            q, i, _ = to_masked_qaoa(p, depth=d)

            _, p = q(i)
            res.append(p)
        plt.plot(range(3, 15), res, linewidth=3, label=f"depth={d}")

    plt.legend()
    plt.xticks(range(3, 15, 2), range(3, 15, 2))
    plt.xlabel("Ancilla Qubits $M$")
    plt.ylabel("Success Probability")
    plt.savefig(f"plots/prob_num_qubits.pdf", bbox_inches="tight", transparent=True)


def plot_prob_depth_example1():
    sns.set(palette="deep", font_scale=1.5, style="ticks")
    plt.subplots(figsize=(5, 4))
    res = []
    for d in range(2, 50, 4):
        print(d)
        p, _, _ = example1(n2=4)
        q, i, _ = to_masked_qaoa(p, depth=d, exact=True)

        _, p = q(i)
        res.append(p)
    plt.plot(
        range(2, 50, 4),
        res,
        linewidth=2,
        label=f"exact",
        color="gray",
        linestyle="dashed",
    )
    for j in [4, 8, 12]:
        res = []
        for d in range(2, 50, 4):
            print(d)
            p, _, _ = example1(n2=j)
            q, i, _ = to_masked_qaoa(p, depth=d)

            _, p = q(i)
            res.append(p)
        plt.plot(range(2, 50, 4), res, linewidth=3, label=f"ancilla={j}")
    plt.legend()
    plt.xticks(range(0, 50, 10), range(0, 50, 10))
    plt.xlabel("QAOA depth $p$")
    plt.ylabel("Success Probability")
    plt.savefig(f"plots/prob_depth.pdf", bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    # plot_inequality()
    # plot_example(example2)
    opt_example(example2, typ="penalty", anc=12)
    # plot_prob_depth_example1()
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
