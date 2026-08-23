"""Base interface for per-pixel selector score components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any

from ..types import SelectorInput


class ScoreComponent(ABC):
    """Produce one term of the composite degradation score map.

    Concrete implementations include predictive entropy, prior-weighted
    uncertainty, local boundary disagreement, and the conditional frequency
    residual described in the methodology.
    """

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = float(weight)

    @abstractmethod
    def compute(self, inputs: SelectorInput) -> Tensor:
        """Return a score map with spatial shape ``(B, H, W)``."""

    def __call__(self, inputs: SelectorInput) -> Tensor:
        return self.compute(inputs)
