from typing import Callable, Optional, Tuple, TypeVar

import jax
import jax.numpy as jnp

import diffrax
from diffrax.custom_types import Bool, DenseInfo, PyTree, PyTreeDef, Scalar
from diffrax.term import AbstractTerm
from diffrax.solution import RESULTS

SolverState = TypeVar("SolverState", bound=Optional[PyTree])


class InBoundsSolver(diffrax.AbstractSolver):
    solver: diffrax.AbstractSolver
    out_of_bounds: Callable
        
    def __init__(self, solver, out_of_bounds):
        self.solver = solver
        self.out_of_bounds = out_of_bounds
        
    @property
    def term_structure(self):
        return self.solver.term_structure

    @property
    def interpolation_cls(self):
        return self.solver.interpolation_cls

    def order(self, terms: PyTree[AbstractTerm]) -> Optional[int]:
        return self.solver.order(terms)

    def strong_order(self, terms: PyTree[AbstractTerm]) -> Optional[Scalar]:
        return self.solver.strong_order(terms)

    def error_order(self, terms: PyTree[AbstractTerm]) -> Optional[Scalar]:
        order = self.order(terms)
        if order is not None:
            order = order + 1
        return order

    
    def func(self, terms: PyTree[AbstractTerm], t0: Scalar, y0: PyTree, args: PyTree):
        return self.solver.func(terms, t0, y0, args)

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
    ):
        return self.solver.init(terms, t0, t1, y0, args)

    
    def step(
        self,
        terms: PyTree[AbstractTerm],
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
        solver_state: SolverState,
        made_jump: Bool,
    ) -> Tuple[PyTree, Optional[PyTree], DenseInfo, SolverState, RESULTS]:

        y1, y_error, dense_info, solver_state, result = self.solver.step(
            terms, t0, t1, y0, args, solver_state, made_jump
        )
        
        oob = self.out_of_bounds(y1)
        keep = lambda y: jnp.where(oob, jnp.inf, y)
        y_error = jax.tree_util.tree_map(keep, y_error)

        return y1, y_error, dense_info, solver_state, result
