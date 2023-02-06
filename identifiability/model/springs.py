import jax
import jax.numpy as jnp

import diffrax

from .base import AbstractModel
from .. import diffrax_utils
from .. import signals


y0_default = jnp.array([-1.0, 0.0, 1.0, 0.0, 0.0, 0.0])


class SpringsModel(AbstractModel):
    parameters_names = [ 
        'm1', 'm2', 'm3',  # masses
        'b1', 'b2', 'b3',  # damping
        'k1', 'k2',        # spring stiffness
        'l1', 'l2',        # spring lengths
    ]

    def __init__(
        self,
        parameters_default,
        ts,
        signal,
        y0=y0_default,
        diffrax_solver=diffrax_utils.DEFAULT_SOLVER,
        diffrax_stepsize_controller=diffrax_utils.DEFAULT_STEPSIZE_CONTROLLER,
        diffrax_saveat=None,
        diffrax_max_steps=1<<12,
    ):
        self.parameters_default = parameters_default
        self.signal = signal
        self.ts = ts
        self.y0 = y0

        self.diffrax_solver = diffrax_solver
        self.diffrax_stepsize_controller = diffrax_stepsize_controller
        if diffrax_saveat is None:
            self.diffrax_saveat = diffrax.SaveAt(ts=ts) 
        else:
            self.diffrax_saveat = diffrax_saveat
        self.diffrax_max_steps = diffrax_max_steps

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
            p1, p2, p3, v1, v2, v3 = y
            m1, m2, m3, b1, b2, b3, k1, k2, l1, l2 = args

            f12 = (p2 - p1 - l1) * k1
            f23 = (p3 - p2 - l2) * k2

            a1 = (  f12       - v1 * b1 + self.signal(t)) / m1
            a2 = (  f23 - f12 - v2 * b2                 ) / m2
            a3 = (- f23       - v3 * b3                 ) / m3

            return jnp.array([
                v1, v2, v3,
                a1, a2, a3,
            ])

        return derivative_fn

    def solve_ode(self, parameters):
        term = diffrax.ODETerm(self._make_derivative_fn())
        solution = diffrax.diffeqsolve(
            term,
            self.diffrax_solver,
            t0=self.ts[0], t1=self.ts[-1], dt0=0.01,
            y0=self.y0,
            args=parameters,
            saveat=self.diffrax_saveat,
            stepsize_controller=self.diffrax_stepsize_controller,
            max_steps=self.diffrax_max_steps,
            adjoint=diffrax.NoAdjoint(),
            throw=False,
        )

        # report parameters if ODE integration failed
        result = solution.result
        jax.lax.cond(
            result,
            lambda: jax.debug.print(
                "🤯 parameters={parameters}, result={result}",
                parameters=parameters, result=result
            ),
            lambda: None,
        )

        return solution


parameters_default = jnp.array([
    2.0, 9.0, 7.0,  # masses
    0.5, 0.5, 0.5,  # damping
    0.3, 0.3,       # spring stiffness
    1.0, 1.0,       # spring lengths
])

springs_train = SpringsModel(
    parameters_default=parameters_default,
    ts=jnp.linspace(0, 40, 41),
    signal=lambda t: 0.2 * (t < 10),
)

springs_test = SpringsModel(
    parameters_default=parameters_default,
    ts=jnp.linspace(0, 180, 301),
    signal=lambda t: 0.2 * jnp.sin(t * jnp.pi / 20),
)
