"""Budget-constrained region allocation interfaces."""

from .base import RegionAllocator
from .score_area_knapsack import ScoreAreaKnapsackAllocator

__all__ = ["RegionAllocator", "ScoreAreaKnapsackAllocator"]
