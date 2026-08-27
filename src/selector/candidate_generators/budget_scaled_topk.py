"""Per-tile candidate generation scaled from the refinement budget."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..types import SelectionBudget, SelectorInput
from .base import CandidateGenerator


class BudgetScaledTopKCandidateGenerator(CandidateGenerator):
    r"""Select an exact Top-N candidate pool independently for every tile.

    The candidate count is derived from the final per-tile refinement budget::

        candidate_pixels = ceil(pool_multiplier * refinement_pixels)

    ``pool_multiplier`` is the methodology's :math:`\lambda`. Its default value
    of ``2.0`` gives the region grouper and allocator twice as many candidate
    pixels as the final refinement stage may select. The count is capped at the
    number of pixels in one tile.

    Ties at the cutoff are resolved by flattened spatial order. This preserves
    an exact candidate count without requiring a full sort of every score map.
    """

    def __init__(self, *, pool_multiplier: float = 2.0) -> None:
        try:
            pool_multiplier = float(pool_multiplier)
        except (TypeError, ValueError) as error:
            raise TypeError("pool_multiplier must be a real number.") from error

        if not math.isfinite(pool_multiplier):
            raise ValueError("pool_multiplier must be finite.")
        if pool_multiplier < 1.0:
            raise ValueError(
                "pool_multiplier must be at least 1.0 so the candidate pool "
                "can cover the final refinement budget."
            )

        self.pool_multiplier = pool_multiplier

    def generate(
        self,
        score_map: Tensor,
        inputs: SelectorInput,
        budget: SelectionBudget,
    ) -> Tensor:
        """Return a boolean candidate mask with shape ``(B, H, W)``.

        Candidate ranking is performed separately for each batch item. A
        fractional or absolute ``SelectionBudget`` is interpreted per tile.
        ``inputs`` is accepted for the common candidate-generator interface and
        is reserved for policies that require additional backbone outputs.
        """

        del inputs
        self._validate_score_map(score_map)
        if not isinstance(budget, SelectionBudget):
            raise TypeError("budget must be a SelectionBudget.")

        _, height, width = score_map.shape
        pixels_per_tile = height * width
        refinement_pixels = budget.resolve(pixels_per_tile)
        scaled_budget = self.pool_multiplier * refinement_pixels
        candidate_pixels = (
            pixels_per_tile
            if scaled_budget >= pixels_per_tile
            else math.ceil(scaled_budget)
        )

        candidate_mask = torch.zeros_like(score_map, dtype=torch.bool)
        if candidate_pixels == 0:
            return candidate_mask
        if candidate_pixels == pixels_per_tile:
            return torch.ones_like(score_map, dtype=torch.bool)

        flat_scores = score_map.reshape(score_map.shape[0], pixels_per_tile)
        top_values = torch.topk(
            flat_scores,
            k=candidate_pixels,
            dim=1,
            largest=True,
            sorted=False,
        ).values
        thresholds = top_values.min(dim=1, keepdim=True).values

        # Scores strictly above the cutoff always qualify. If several scores
        # equal the cutoff, take only the first required pixels in flattened
        # spatial order so every tile contains exactly candidate_pixels entries.
        above_cutoff = flat_scores > thresholds
        remaining = candidate_pixels - above_cutoff.sum(dim=1, keepdim=True)
        at_cutoff = flat_scores == thresholds
        cutoff_ranks = at_cutoff.cumsum(dim=1)
        selected_at_cutoff = at_cutoff & (cutoff_ranks <= remaining)

        return (above_cutoff | selected_at_cutoff).reshape_as(candidate_mask)

    @staticmethod
    def _validate_score_map(score_map: Tensor) -> None:
        if not isinstance(score_map, Tensor):
            raise TypeError("score_map must be a torch.Tensor.")
        if score_map.ndim != 3:
            raise ValueError(
                "score_map must have shape (B, H, W); "
                f"received {tuple(score_map.shape)}."
            )
        if any(size <= 0 for size in score_map.shape):
            raise ValueError("Every score_map dimension must be non-empty.")
        if not score_map.is_floating_point():
            raise TypeError("score_map must use a floating-point dtype.")
        if not bool(torch.isfinite(score_map).all()):
            raise ValueError("score_map must contain only finite values.")
