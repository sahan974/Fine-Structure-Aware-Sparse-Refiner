"""Base interface for converting scores into candidate pixels."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any

from ..types import SelectionBudget, SelectorInput


class CandidateGenerator(ABC):
    """Convert a continuous score map into a binary candidate mask."""

    @abstractmethod
    def generate(
        self,
        score_map: Tensor,
        inputs: SelectorInput,
        budget: SelectionBudget,
    ) -> Tensor:
        """Return a boolean mask with the same spatial shape as ``score_map``."""

    def __call__(
        self,
        score_map: Tensor,
        inputs: SelectorInput,
        budget: SelectionBudget,
    ) -> Tensor:
        return self.generate(score_map, inputs, budget)
