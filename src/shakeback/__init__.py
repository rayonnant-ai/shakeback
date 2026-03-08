"""Shakeback: unstick backpropagation with noise perturbation."""

__version__ = "0.1.0"

from shakeback.core import shakeback
from shakeback.problem import Problem

__all__ = ["shakeback", "Problem"]
