"""Base interface for selecting complete regions under a pixel budget."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ..types import AllocationResult, Region, SelectionBudget, SelectorInput


class RegionAllocator(ABC):
    """Choose regions under a budget and optional class-stratum quotas."""

    @abstractmethod
    def allocate(
        self,
        regions: Sequence[Region],
        budget: SelectionBudget,
        inputs: SelectorInput,
    ) -> AllocationResult:
        """Return selected regions and their final boolean selection mask."""

    def __call__(
        self,
        regions: Sequence[Region],
        budget: SelectionBudget,
        inputs: SelectorInput,
    ) -> AllocationResult:
        return self.allocate(regions, budget, inputs)
