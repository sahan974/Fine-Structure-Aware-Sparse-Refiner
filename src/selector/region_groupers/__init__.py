"""Candidate-region grouping interfaces and implementations."""

from .base import RegionGrouper
from .spatial_connected_components import SpatialConnectedComponentGrouper

__all__ = ["RegionGrouper", "SpatialConnectedComponentGrouper"]
