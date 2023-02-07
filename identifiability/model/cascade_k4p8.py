import jax.numpy as jnp

from .cascade_base import CascadeModel, prior_log_prob_base
from .. import signals


class CascadeK4P8(CascadeModel):
    parameters_names = [ 
        'a1', 'd1',
        'a2', 'd2',
        'a3', 'd3',
        'a4', 'd4',
    ]

    @property
    def y_dim(self):
        return 4

    def _make_derivative_fn(self):
        def derivative_fn(t, y, args):
            a1, d1, a2, d2, a3, d3, a4, d4 = args
            y0 = self.signal(t)
            y1, y2, y3, y4 = y

            return jnp.array([
                a1 * y0 * (1 - y1) - d1 * y1,
                a2 * y1 * (1 - y2) - d2 * y2,
                a3 * y2 * (1 - y3) - d3 * y3,
                a4 * y3 * (1 - y4) - d4 * y4,
            ])

        return derivative_fn

    def prior_log_prob(self, parameters):
        return prior_log_prob_base(parameters)

    def pretty_print_params(self, parameters):
        i = 0
        for section_len in [2, 2, 2, 2]:
            for _ in range(section_len):
                print(f"{self.parameters_names[i]}: {parameters[i]: .2f}", end='  ')
                i += 1
            print()



parameters_default = jnp.array([
    2.0, 8.0,  # a1, d1
    3.0, 3.0,  # a2, d2
    4.0, 4.0,  # a3, d3
   10.0, 2.0,  # a4, d4
])

cascade_k4p8_train = CascadeK4P8(
    ts=jnp.linspace(0, 11, 12),
    signal=signals.make_pulse(4),
)

cascade_k4p8_test_blocky = CascadeK4P8(
    ts=jnp.linspace(0, 12, 121),
    signal=signals.test_blocky,
)

cascade_k4p8_test_wiggly = CascadeK4P8(
    ts=jnp.linspace(0, 12, 121),
    signal=signals.test_wiggly,
)
