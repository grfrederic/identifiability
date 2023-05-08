import jax
import jax.numpy as jnp

from .base import AbstractODEModel
from .. import signals


y0_default = jnp.array([1.0, 0.0, -1.0, 0.0, 0.0, 0.0])


class SpringsModel(AbstractODEModel):
    @property
    def parameters_names(self):
        return [ 
            'm1', 'm2', 'm3',  # masses
            'b1', 'b2', 'b3',  # damping
            'k1', 'k2',        # spring stiffness
        ]

    @property
    def y_dim(self):
        return 3

    def __init__(
        self,
        ts,
        signal,
        y0=y0_default,
        **kwargs,
    ):
        super().__init__(
            ts=ts, signal=signal, y0=y0,
            **kwargs,
        )

    def prior_log_prob(self, parameters):
        ms = parameters[:3]
        rs = parameters[3:]

        ok = (ms < 0.1).sum() + (ms > 20.0).sum() \
           + (rs < 0.1).sum() + (rs >  5.0).sum() == 0

        return jnp.where(ok, 0, -jnp.inf)

    def run(self, parameters):
        solution = self.solve_ode(parameters)
        return solution.ys[..., :3]

    def pretty_print_params(self, parameters):
        i = 0
        for section_len in [3, 3, 2, 2]:
            for _ in range(section_len):
                print(f"{self.parameters_names[i]}: {parameters[i]: .2f}", end='  ')
                i += 1
            print()

    def _make_derivative_fn(self):
        def derivative_fn(t, y, args):
            x1, x2, x3, v1, v2, v3 = y
            m1, m2, m3, b1, b2, b3, k1, k2 = args

            f12 = (x2 - x1 + 1.) * k1  # string lengths = 1.
            f23 = (x3 - x2 + 1.) * k2

            a1 = (  f12       - v1 * b1 + self.signal(t)) / m1
            a2 = (  f23 - f12 - v2 * b2                 ) / m2
            a3 = (- f23       - v3 * b3                 ) / m3

            return jnp.array([
                v1, v2, v3,
                a1, a2, a3,
            ])

        return derivative_fn


parameters_default = jnp.array([
    2.0, 9.0, 7.0,  # masses
    0.5, 0.5, 0.5,  # damping
    0.3, 0.3,       # spring stiffness
])

parameters_median = jnp.array([
    10.05, 10.05, 10.05,  # masses
    2.55, 2.55, 2.55,     # damping
    2.55, 2.55,           # spring stiffness
])
