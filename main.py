from typing import *
import time
from dataclasses import dataclass
from functools import cached_property
import numpy as np

import jax.numpy as jnp
import jax

import matplotlib.pyplot as plt


@dataclass
class LinSpace:
    space: Tuple[float, float]
    num_qubits: int

    @property
    def shape(self):
        return 2**self.num_qubits

    @cached_property
    def array(self):
        return np.linspace(*self.space, self.shape)


@dataclass
class Domain:
    vars: List[LinSpace]

    @property
    def num_qubits(self) -> int:
        return sum(var.num_qubits for var in self.vars)

    @property
    def shape(self) -> Tuple:
        return tuple(var.shape for var in self.vars)

    @cached_property
    def array(self):
        res = np.meshgrid(*(var.array for var in self.vars))
        return jnp.array(np.array([r.reshape((-1,)) for r in res[::-1]]).T)


@dataclass
class Constraint:
    domain: Domain
    lhs: Callable[[Any], float]
    rhs: float
    num_qubits: int

    @cached_property
    def lhs_values(self):
        f = jax.jit(jax.vmap(self.lhs, in_axes=(0,)))
        return f(self.domain.array)


@dataclass
class Problem:
    domain: Domain
    obj: Callable[[Any], float]
    constraints: List[Constraint]

    @cached_property
    def obj_values(self):
        f = jax.jit(jax.vmap(self.obj, in_axes=(0,)))
        return f(self.domain.array)

    def eval(self):
        for c in self.constraints:
            c.lhs_values
        self.obj_values


if __name__ == "__main__":
    a = LinSpace(space=(-1.0, 1.0), num_qubits=10)
    domain = Domain([a, a])

    c = Constraint(domain, lhs=lambda x: x[0] + x[1], rhs=0, num_qubits=4)

    p = Problem(
        domain,
        lambda x: 0.5 * (x[0] - 0.4) ** 2 + (x[1] - 0.3) ** 2,
        constraints=[c],
    )
    a = time.time()
    p.eval()
    print(time.time() - a)
