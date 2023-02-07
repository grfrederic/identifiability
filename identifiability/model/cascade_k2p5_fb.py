import jax.numpy as jnp

from .cascade_base import CascadeModel, prior_log_prob_base
from .. import signals


class CascadeK2P5Fb(CascadeModel):
    parameters_names = [ 
        'a1', 'd1',
        'a2', 'd2',
        'fb',
    ]

    @property
    def y_dim(self):
        return 2

    def _make_derivative_fn(self):
        def derivative_fn(t, y, args):
            a1, d1, a2, d2, fb = args
            y0 = self.signal(t)
            y1, y2 = y

            return jnp.array([
                a1 * y0 / (1 + fb * y2) * (1 - y1) - d1 * y1,
                a2 * y1                 * (1 - y2) - d2 * y2,
            ])

        return derivative_fn

    def prior_log_prob(self, parameters):
        parameters_normalized = (
            parameters / jnp.array([1., 1., 1., 1., 1000.])
        )
        return prior_log_prob_base(parameters_normalized)

    def pretty_print_params(self, parameters):
        i = 0
        for section_len in [2, 2, 1]:
            for _ in range(section_len):
                print(f"{self.parameters_names[i]}: {parameters[i]: .2f}", end='  ')
                i += 1
            print()



parameters_default = jnp.array([
   10.0, 2.0,  # a1, d1
   10.0, 2.0,  # a2, d2
      1000.0,  # fb
])

cascade_k2p5_fb_train = CascadeK2P5Fb(
    ts=jnp.linspace(0, 11, 12),
    signal=signals.make_pulse(4),
)

cascade_k2p5_fb_test_blocky = CascadeK2P5Fb(
    ts=jnp.linspace(0, 12, 121),
    signal=signals.test_blocky,
)

cascade_k2p5_fb_test_wiggly = CascadeK2P5Fb(
    ts=jnp.linspace(0, 12, 121),
    signal=signals.test_wiggly,
)
