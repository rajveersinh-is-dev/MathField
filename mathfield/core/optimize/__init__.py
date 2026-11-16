"""
mathfield/core/optimize/

Stage 4: Joint Optimization Engine.
"""

from mathfield.core.optimize.conditioning import ConditionMonitor, ConditionReport
from mathfield.core.optimize.evolutionary import EvolutionaryOptimizer
from mathfield.core.optimize.solvers import (
    LBFGSBOptimizer,
    LevenbergMarquardtOptimizer,
    solve_linear_lstsq,
)
from mathfield.core.optimize.varpro import VarProOptimizer, VarProResult

__all__ = [
    "VarProOptimizer",
    "VarProResult",
    "LevenbergMarquardtOptimizer",
    "LBFGSBOptimizer",
    "solve_linear_lstsq",
    "EvolutionaryOptimizer",
    "ConditionMonitor",
    "ConditionReport",
]
