"""Shared contracts used by every selector stage.

The project will use PyTorch tensors, but importing this module does not require
PyTorch to be installed.  ``Tensor`` is therefore used as a type-only name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any


@dataclass(frozen=True)
class SelectorInput:
    """Inputs produced by a frozen segmentation backbone.

    Attributes:
        probabilities: Per-class posterior with shape ``(B, K, H, W)``.
        image: Raw input image with shape ``(B, C, H, W)`` when required.
        features: Named intermediate feature maps from the backbone.
        class_frequencies: Training-set pixel prior for each class.
        metadata: Optional sample information that selector strategies may use.
    """

    probabilities: Tensor
    image: Tensor | None = None
    features: Mapping[str, Tensor] = field(default_factory=dict)
    class_frequencies: Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Region:
    """One candidate component proposed for refinement.

    ``region_id`` is unique within one selector call. ``batch_index`` identifies
    the source tile, and ``pixel_indices`` contains flattened ``H * W`` indices
    relative to that tile.
    """

    region_id: int
    batch_index: int
    pixel_indices: Tensor
    aggregate_score: float
    area: int
    dominant_class: int | None = None
    class_stratum: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectionBudget:
    """A refinement budget expressed as a fraction or an exact pixel count."""

    fraction: float | None = None
    max_pixels: int | None = None

    def __post_init__(self) -> None:
        if (self.fraction is None) == (self.max_pixels is None):
            raise ValueError("Set exactly one of fraction or max_pixels.")
        if self.fraction is not None and not 0.0 <= self.fraction <= 1.0:
            raise ValueError("fraction must be between 0 and 1.")
        if self.max_pixels is not None and self.max_pixels < 0:
            raise ValueError("max_pixels must be non-negative.")

    def resolve(self, total_pixels: int) -> int:
        """Return the maximum number of selectable pixels."""

        if total_pixels < 0:
            raise ValueError("total_pixels must be non-negative.")
        if self.max_pixels is not None:
            return min(self.max_pixels, total_pixels)
        return min(int(total_pixels * self.fraction), total_pixels)  # type: ignore[arg-type]


@dataclass(frozen=True)
class AllocationResult:
    """Output of a region allocator."""

    selected_mask: Tensor
    selected_regions: Sequence[Region]
    used_pixels: int
    budget_pixels: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectionResult:
    """Complete, inspectable output of the selector pipeline."""

    score_map: Tensor
    candidate_mask: Tensor
    regions: Sequence[Region]
    allocation: AllocationResult
