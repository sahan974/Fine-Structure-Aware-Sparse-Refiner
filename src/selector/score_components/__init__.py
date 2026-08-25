"""Per-pixel degradation score interfaces."""

from .base import ScoreComponent
from .local_boundary_disagreement import LocalBoundaryDisagreement
from .prior_weighted_uncertainty import PriorWeightedUncertainty

__all__ = [
    "LocalBoundaryDisagreement",
    "PriorWeightedUncertainty",
    "ScoreComponent",
]
