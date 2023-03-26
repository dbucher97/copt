from typing import *
from dataclasses import dataclass
from functools import cached_property
import numpy as np

import jax.numpy as jnp
import jax

@dataclass
class LinSpace:
    space: Tuple[float, float]
    num_qubits: int
    endpoint: bool = True

    @property
    def shape(self):
        return 2**self.num_qubits

    @cached_property
    def array(self):
        return np.linspace(*self.space, self.shape, endpoint=self.endpoint)


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


def get_prob(n):
    x = np.arange(2 ** (n - 1))

    @np.vectorize
    def prob(t):
        d = np.pi * (x - 2**n * t)
        s1 = np.sin(d)
        s2 = np.sin(d * 2 ** (-n))
        ic = np.isclose(d, 0)
        v = np.abs(np.divide(s1, s2, where=np.logical_not(ic))) ** 2
        return (np.sum(v)) / 2 ** (2 * n) + np.sum(ic)

    return prob


@dataclass
class Constraint:
    domain: Domain
    lhs: Callable[[Any], float]
    rhs: float
    num_qubits: int

    underdrive: float = 0.9
    pint: int = 3

    @cached_property
    def lhs_values(self):
        f = jax.jit(jax.vmap(self.lhs, in_axes=(0,)))
        return f(self.domain.array)

    @cached_property
    def penalty_tesselation(self):
        scale = 2**self.num_qubits / (self.rhs - jnp.min(self.lhs_values))
        scaled_lhs_values = (self.lhs_values - self.rhs) * scale
        masked = jnp.minimum(scaled_lhs_values, 0)
        return ((scaled_lhs_values - jnp.round(masked)) / scale) ** 2

    @cached_property
    def constrained_measure(self):
        n = self.num_qubits
        possible = np.linspace(0, -1, 2 ** (n - 1), endpoint=False)

        scale = self.underdrive / max(
            self.rhs - jnp.min(self.lhs_values), jnp.max(self.lhs_values) - self.rhs
        )
        values = (self.lhs_values - self.rhs) * scale

        inpx = jnp.linspace(-1, 1, 2 ** (n + self.pint), endpoint=False)


        def fun(val):
            x = (val - possible) * jnp.pi

            ic = jnp.isclose(x, 0)

            result = jnp.where(
                ic, 1.0, (jnp.sin(2 ** (n - 1) * x) / jnp.sin(x / 2) / 2**n) ** 2
            ).sum()

            return result

        f2 = jax.jit(jax.vmap(fun, in_axes=(0,)))
        inpy = f2(inpx)

        return jnp.interp(values, inpx, inpy)

    @cached_property
    def feasible(self):
        return self.lhs_values <= self.rhs


@dataclass
class Problem:
    domain: Domain
    obj: Callable[[Any], float]
    constraints: List[Constraint]

    @cached_property
    def obj_values(self):
        f = jax.jit(jax.vmap(self.obj, in_axes=(0,)))
        return f(self.domain.array)

    @cached_property
    def feasible(self):
        res = jnp.ones(2 ** self.domain.num_qubits, dtype=bool)
        for c in self.constraints:
            res = jnp.logical_and(c.feasible, res)
        return res

    def eval(self):
        for c in self.constraints:
            c.lhs_values
        self.obj_values

    def get_min(self):
        m = jnp.argmin(jnp.where(self.feasible, self.obj_values, jnp.inf))
        v = self.obj_values[m]
        idx = jnp.unravel_index(m, self.domain.shape)
        return v, idx

