import time

import matplotlib.pyplot as plt

from copt.objects import LinSpace, Constraint, Problem, Domain
from copt.qaoa import to_penalty_qaoa


if __name__ == "__main__":
    a = LinSpace(space=(-1.0, 1.0), num_qubits=8)
    domain = Domain([a, a])

    c = Constraint(domain, lhs=lambda x: x[0] + x[1], rhs=0, num_qubits=8)

    p = Problem(
        domain,
        lambda x: 0.5 * (x[0] - 0.4) ** 2 + (x[1] - 0.3) ** 2,
        constraints=[c],
    )

    x = to_penalty_qaoa(p, 10)

    plt.matshow(x.reshape(domain.shape), norm="log")
    plt.show()
