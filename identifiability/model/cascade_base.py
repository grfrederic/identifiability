from abc import abstractmethod
import jax
import jax.numpy as jnp

import diffrax

from .base import AbstractModel
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


class CascadeModel(AbstractModel):
    def __init__(
        self,
        ts,
        signal,
        y0=None,
        diffrax_solver=DEFAULT_CASCADE_SOLVER,
        diffrax_stepsize_controller=DEFAULT_CASCADE_STEPSIZE_CONTROLLER,
        diffrax_saveat=None,
        diffrax_max_steps=DEFAULT_CASCADE_MAX_STEPS,
    ):
        self.signal = signal
        self.ts = ts

        if y0 is None:
            self.y0 = jnp.zeros(self.y_dim)
        else:
            self.y0 = y0

        self.diffrax_solver = diffrax_solver
        self.diffrax_stepsize_controller = diffrax_stepsize_controller
        if diffrax_saveat is None:
            self.diffrax_saveat = diffrax.SaveAt(ts=ts) 
        else:
            self.diffrax_saveat = diffrax_saveat
        self.diffrax_max_steps = diffrax_max_steps

    @abstractmethod
    def _make_derivative_fn(self):
        pass

    @property
    @abstractmethod
    def y_dim(self):
        pass

    def run(self, parameters):
        solution = self.solve_ode(parameters)
        return softlog(solution.ys)

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
            discrete_terminating_event=diffrax.DiscreteTerminatingEvent(stop),
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
