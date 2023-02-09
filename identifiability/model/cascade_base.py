from abc import abstractmethod
import jax
import jax.numpy as jnp

import diffrax

from .base import AbstractODEModel
from ..in_bounds_solver import InBoundsSolver
from .. import diffrax_utils
from .. import signals


DEFAULT_CASCADE_SOLVER = InBoundsSolver(
    diffrax.Dopri5(),
    lambda y: (y < 0.0) | (y > 1.0),
)
DEFAULT_CASCADE_MAX_STEPS = 1<<20


# let solver know to watch out at integer times that signal might change
DEFAULT_CASCADE_STEPSIZE_CONTROLLER = diffrax.PIDController(
    **diffrax_utils.DEFAULT_STEPSIZE_CONTROLLER_KWARGS,
    jump_ts=jnp.linspace(0, 24, 25),
)


# stopping condition for solver, perhaps not needed
def bad_y(y):
    return jnp.isnan(y).any() \
         + (y <= -0.01).any() \
         + (y >= +1.01).any()

def stop(state, *args, **kwargs):
    return bad_y(state.y)


# applied to trajectories before calculating errors
def softlog(x):
    return jnp.log10(0.01 + x)


# used as a basis for construcing priors
def prior_log_prob_base(parameters):
    ok = (parameters > 10000).sum() == 0

    log_parameters = jnp.log(parameters)
    log_prob = -0.5 * jnp.square((log_parameters - 1.0) / 3.0).sum()

    return jnp.where(ok, log_prob, -jnp.inf)


class CascadeModel(AbstractODEModel):
    def __init__(
        self,
        ts,
        signal,
        y0=None,
        diffrax_solver=DEFAULT_CASCADE_SOLVER,
        diffrax_diffeqsolve_kwargs=dict(
            stepsize_controller=DEFAULT_CASCADE_STEPSIZE_CONTROLLER,
            max_steps=DEFAULT_CASCADE_MAX_STEPS,
            discrete_terminating_event=diffrax.DiscreteTerminatingEvent(stop),
        )
    ):
        super().__init__(
            ts=ts, signal=signal, y0=y0,
            diffrax_solver=diffrax_solver,
            diffrax_diffeqsolve_kwargs=diffrax_diffeqsolve_kwargs,
        )

    def run(self, parameters):
        solution = self.solve_ode(parameters)
        return softlog(solution.ys)
