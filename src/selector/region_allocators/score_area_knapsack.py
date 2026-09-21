"""Quota-free complete-region allocation under a pixel budget."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from ..types import AllocationResult, Region, SelectionBudget, SelectorInput
from .base import RegionAllocator


@dataclass(frozen=True)
class _Decision:
    """One persistent back-pointer in a knapsack solution."""

    region: Region
    previous: _Decision | None


@dataclass(frozen=True)
class _KnapsackState:
    """Best known value for one exact used-pixel count."""

    score: float
    decision: _Decision | None


class ScoreAreaKnapsackAllocator(RegionAllocator):
    """Select complete high-value regions without frequency quotas.

    Each tile is allocated independently. A region's ``aggregate_score`` is
    its value and its ``area`` is its pixel cost. The allocator maximizes the
    sum of selected region scores while keeping the total selected area at or
    below the tile's refinement budget.

    The implementation uses an exact, Pareto-pruned dynamic program. Its
    frontier stores only non-dominated ``(used_pixels, score)`` states: a
    state is removed whenever another state uses no more pixels and has an
    equal or higher score. This preserves the 0-1 knapsack optimum while
    avoiding a dense ``regions x budget`` table for typical sparse inputs.

    Regions are indivisible. A region larger than the tile budget is skipped;
    the allocator never truncates it to consume remaining capacity. Ties are
    deterministic because regions are considered in ascending ``region_id``
    order and an existing equal-score state is retained.
    """

    _SCORE_TOLERANCE = 1e-12

    def allocate(
        self,
        regions: Sequence[Region],
        budget: SelectionBudget,
        inputs: SelectorInput,
    ) -> AllocationResult:
        """Return the exact quota-free allocation for every batch tile."""

        height, width, batch_size, device = self._validate_inputs(
            regions=regions,
            budget=budget,
            inputs=inputs,
        )
        pixels_per_tile = height * width
        budget_per_tile = budget.resolve(pixels_per_tile)

        regions_by_tile: list[list[Region]] = [
            [] for _ in range(batch_size)
        ]
        for region in regions:
            self._validate_region(
                region=region,
                batch_size=batch_size,
                pixels_per_tile=pixels_per_tile,
                device=device,
            )
            regions_by_tile[region.batch_index].append(region)

        selected_mask = torch.zeros(
            (batch_size, height, width),
            dtype=torch.bool,
            device=device,
        )
        selected_regions: list[Region] = []
        per_tile_metadata: list[dict[str, object]] = []

        for batch_index, tile_regions in enumerate(regions_by_tile):
            selected_tile_regions, oversized_region_ids = self._solve_tile(
                tile_regions=tile_regions,
                budget_pixels=budget_per_tile,
            )
            selected_regions.extend(selected_tile_regions)

            for region in selected_tile_regions:
                selected_mask[batch_index].reshape(-1)[
                    region.pixel_indices
                ] = True

            used_pixels = sum(region.area for region in selected_tile_regions)
            per_tile_metadata.append(
                {
                    "batch_index": batch_index,
                    "budget_pixels": budget_per_tile,
                    "used_pixels": used_pixels,
                    "candidate_region_count": len(tile_regions),
                    "selected_region_ids": tuple(
                        region.region_id for region in selected_tile_regions
                    ),
                    "oversized_region_ids": tuple(oversized_region_ids),
                }
            )

        used_pixels = sum(region.area for region in selected_regions)
        selected_pixels = int(selected_mask.sum().item())
        if selected_pixels != used_pixels:
            raise ValueError(
                "Regions selected for allocation overlap. The region grouper "
                "must return disjoint pixel sets within each batch tile."
            )

        return AllocationResult(
            selected_mask=selected_mask,
            selected_regions=tuple(selected_regions),
            used_pixels=used_pixels,
            budget_pixels=budget_per_tile * batch_size,
            metadata={
                "allocator": "score_area_knapsack",
                "quota_applied": False,
                "per_tile": tuple(per_tile_metadata),
            },
        )

    def _solve_tile(
        self,
        *,
        tile_regions: Sequence[Region],
        budget_pixels: int,
    ) -> tuple[list[Region], list[int]]:
        """Solve one exact 0-1 score-area knapsack problem."""

        eligible_regions: list[Region] = []
        oversized_region_ids: list[int] = []
        for region in sorted(tile_regions, key=lambda item: item.region_id):
            if region.area > budget_pixels:
                oversized_region_ids.append(region.region_id)
            else:
                eligible_regions.append(region)

        # Key: exact used-pixel count. Value: highest score at that count.
        frontier: dict[int, _KnapsackState] = {
            0: _KnapsackState(score=0.0, decision=None)
        }
        for region in eligible_regions:
            updated_frontier = dict(frontier)
            for used_pixels, state in frontier.items():
                next_used_pixels = used_pixels + region.area
                if next_used_pixels > budget_pixels:
                    continue

                next_score = state.score + region.aggregate_score
                existing = updated_frontier.get(next_used_pixels)
                if (
                    existing is None
                    or next_score > existing.score + self._SCORE_TOLERANCE
                ):
                    updated_frontier[next_used_pixels] = _KnapsackState(
                        score=next_score,
                        decision=_Decision(
                            region=region,
                            previous=state.decision,
                        ),
                    )

            frontier = self._prune_dominated_states(updated_frontier)

        best_state = frontier[0]
        for used_pixels in sorted(frontier):
            state = frontier[used_pixels]
            if state.score > best_state.score + self._SCORE_TOLERANCE:
                best_state = state

        selected_regions: list[Region] = []
        decision = best_state.decision
        while decision is not None:
            selected_regions.append(decision.region)
            decision = decision.previous

        selected_regions.reverse()
        return selected_regions, oversized_region_ids

    def _prune_dominated_states(
        self,
        states: dict[int, _KnapsackState],
    ) -> dict[int, _KnapsackState]:
        """Keep only states that can still improve a future solution."""

        frontier: dict[int, _KnapsackState] = {}
        best_score = -math.inf
        for used_pixels in sorted(states):
            state = states[used_pixels]
            if state.score > best_score + self._SCORE_TOLERANCE:
                frontier[used_pixels] = state
                best_score = state.score
        return frontier

    @staticmethod
    def _validate_inputs(
        *,
        regions: Sequence[Region],
        budget: SelectionBudget,
        inputs: SelectorInput,
    ) -> tuple[int, int, int, torch.device]:
        if not isinstance(inputs, SelectorInput):
            raise TypeError("inputs must be a SelectorInput.")
        if not isinstance(budget, SelectionBudget):
            raise TypeError("budget must be a SelectionBudget.")
        if not isinstance(regions, Sequence):
            raise TypeError("regions must be a sequence of Region objects.")

        probabilities = inputs.probabilities
        if not isinstance(probabilities, Tensor):
            raise TypeError("inputs.probabilities must be a torch.Tensor.")
        if probabilities.ndim != 4:
            raise ValueError(
                "inputs.probabilities must have shape (B, K, H, W); "
                f"received {tuple(probabilities.shape)}."
            )
        if any(size <= 0 for size in probabilities.shape):
            raise ValueError("Every probability tensor dimension must be non-empty.")

        batch_size, _, height, width = probabilities.shape
        return height, width, batch_size, probabilities.device

    @staticmethod
    def _validate_region(
        *,
        region: Region,
        batch_size: int,
        pixels_per_tile: int,
        device: torch.device,
    ) -> None:
        if not isinstance(region, Region):
            raise TypeError("regions must contain only Region objects.")
        if isinstance(region.region_id, bool) or not isinstance(region.region_id, int):
            raise TypeError("region.region_id must be an integer.")
        if (
            isinstance(region.batch_index, bool)
            or not isinstance(region.batch_index, int)
        ):
            raise TypeError("region.batch_index must be an integer.")
        if not 0 <= region.batch_index < batch_size:
            raise ValueError("region.batch_index is outside the input batch.")
        if isinstance(region.area, bool) or not isinstance(region.area, int):
            raise TypeError("region.area must be an integer.")
        if region.area <= 0:
            raise ValueError("region.area must be positive.")
        if not math.isfinite(float(region.aggregate_score)):
            raise ValueError("region.aggregate_score must be finite.")

        pixel_indices = region.pixel_indices
        if not isinstance(pixel_indices, Tensor):
            raise TypeError("region.pixel_indices must be a torch.Tensor.")
        if pixel_indices.ndim != 1:
            raise ValueError("region.pixel_indices must be one-dimensional.")
        if pixel_indices.dtype != torch.long:
            raise TypeError("region.pixel_indices must use torch.long.")
        if pixel_indices.device != device:
            raise ValueError(
                "region.pixel_indices and inputs.probabilities must be on the "
                "same device."
            )
        if pixel_indices.numel() != region.area:
            raise ValueError("region.area must equal the number of pixel indices.")
        if bool((pixel_indices < 0).any()) or bool(
            (pixel_indices >= pixels_per_tile).any()
        ):
            raise ValueError("region.pixel_indices contains an out-of-range value.")
        if pixel_indices.unique().numel() != pixel_indices.numel():
            raise ValueError("region.pixel_indices must not contain duplicates.")
