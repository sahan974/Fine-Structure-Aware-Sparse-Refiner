"""Candidate-mask generation interfaces and implementations."""

from .base import CandidateGenerator
from .budget_scaled_topk import BudgetScaledTopKCandidateGenerator

__all__ = ["BudgetScaledTopKCandidateGenerator", "CandidateGenerator"]
