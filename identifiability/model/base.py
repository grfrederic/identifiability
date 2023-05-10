from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp

import diffrax

from .. import diffrax_utils


class AbstractModel(ABC):
    @abstractmethod
    def run(self):
        """Run the model."""
        pass

    @abstractmethod
    def prior_log_prob(self, parameters):
        """Log-probability of parameters."""
        pass

    @property
    @abstractmethod
    def parameters_names(self):
        pass

    @property
    @abstractmethod
    def y_dim(self):
        pass


class AbstractODEModel(AbstractModel):
    def __init__(
        self,
        ts,
        signal,
        y0=None,
        diffrax_solver=diffrax_utils.DEFAULT_SOLVER,
        diffrax_diffeqsolve_kwargs=dict(
            stepsize_controller=diffrax_utils.DEFAULT_STEPSIZE_CONTROLLER,
            max_steps=1<<12,
        )
    ):
        self.signal = signal
        self.ts = ts
        self.y0 = jnp.zeros(self.y_dim) if y0 is None else y0

        self.diffrax_solver = diffrax_solver

        diffrax_diffeqsolve_kwargs_ = {'saveat': diffrax.SaveAt(ts=ts)} \
                                    | diffrax_diffeqsolve_kwargs

        self.diffrax_diffeqsolve_kwargs = diffrax_diffeqsolve_kwargs_

    @abstractmethod
    def _make_derivative_fn(self):
        pass

    def solve_ode(self, parameters):
        term = diffrax.ODETerm(self._make_derivative_fn())
        solution = diffrax.diffeqsolve(
            term,
            self.diffrax_solver,
            t0=self.ts[0], t1=self.ts[-1], dt0=0.01,
            y0=self.y0,
            args=parameters,
            throw=False,
            **self.diffrax_diffeqsolve_kwargs,
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
