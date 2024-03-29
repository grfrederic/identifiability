import jax.numpy as jnp

from .cascade_base import CascadeModel, prior_log_prob_base
from .. import signals


class CascadeK4P9Fb(CascadeModel):
    @property
    def parameters_names(self):
        return [ 
            'a1', 'd1',
            'a2', 'd2',
            'a3', 'd3',
            'a4', 'd4',
            'f1',
        ]

    @property
    def y_dim(self):
        return 4

    def _make_derivative_fn(self):
        def derivative_fn(t, y, args):
            a1, d1, a2, d2, a3, d3, a4, d4, f1 = args
            y0 = self.signal(t)
            y1, y2, y3, y4 = y

            return jnp.array([
                a1 * y0 / (1 + f1 * y4) * (1 - y1) - d1 * y1,
                a2 * y1                 * (1 - y2) - d2 * y2,
                a3 * y2                 * (1 - y3) - d3 * y3,
                a4 * y3                 * (1 - y4) - d4 * y4,
            ])

        return derivative_fn

    def prior_log_prob(self, parameters):
        parameters_normalized = (
            parameters / jnp.array([1., 1., 1., 1., 1., 1., 1., 1., 1000.])
        )
        return prior_log_prob_base(parameters_normalized)

    def pretty_print_params(self, parameters):
        i = 0
        for section_len in [2, 2, 2, 2, 1]:
            for _ in range(section_len):
                print(f"{self.parameters_names[i]}: {parameters[i]: .2f}", end='  ')
                i += 1
            print()


parameters_default = jnp.array([
    2.0, 8.0,  # a1, d1
    3.0, 3.0,  # a2, d2
    4.0, 4.0,  # a3, d3
   10.0, 2.0,  # a4, d4
      1000.0,  # f1
])


parameters_median = jnp.array([
    jnp.e, jnp.e,  # a1, d1
    jnp.e, jnp.e,  # a2, d2
    jnp.e, jnp.e,  # a3, d3
    jnp.e, jnp.e,  # a4, d4
    1000 * jnp.e,  # f1
])
