"""Base interface for grouping candidate pixels into regions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any

from ..types import Region, SelectorInput


class RegionGrouper(ABC):
    """Form components using spatial and, optionally, frequency coherence."""

    @abstractmethod
    def group(
        self,
        candidate_mask: Tensor,
        score_map: Tensor,
        inputs: SelectorInput,
    ) -> Sequence[Region]:
        """Return region proposals built from candidate pixels."""

    def __call__(
        self,
        candidate_mask: Tensor,
        score_map: Tensor,
        inputs: SelectorInput,
    ) -> Sequence[Region]:
        return self.group(candidate_mask, score_map, inputs)
