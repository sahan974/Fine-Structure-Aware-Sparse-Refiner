"""Composition root for the selector stages."""

from __future__ import annotations

from collections.abc import Sequence

from .candidate_generators.base import CandidateGenerator
from .region_allocators.base import RegionAllocator
from .region_groupers.base import RegionGrouper
from .score_components.base import ScoreComponent
from .types import SelectionBudget, SelectionResult, SelectorInput


class Selector:
    """Compose scoring, candidate generation, grouping, and allocation."""

    def __init__(
        self,
        score_components: Sequence[ScoreComponent],
        candidate_generator: CandidateGenerator,
        region_grouper: RegionGrouper,
        region_allocator: RegionAllocator,
    ) -> None:
        if not score_components:
            raise ValueError("At least one score component is required.")
        self.score_components = tuple(score_components)
        self.candidate_generator = candidate_generator
        self.region_grouper = region_grouper
        self.region_allocator = region_allocator

    def select(
        self,
        inputs: SelectorInput,
        budget: SelectionBudget,
    ) -> SelectionResult:
        """Run the four selector stages and retain their intermediate outputs."""

        score_map = None
        for component in self.score_components:
            weighted_score = component(inputs) * component.weight
            score_map = weighted_score if score_map is None else score_map + weighted_score

        candidate_mask = self.candidate_generator(score_map, inputs)
        regions = self.region_grouper(candidate_mask, score_map, inputs)
        allocation = self.region_allocator(regions, budget, inputs)

        return SelectionResult(
            score_map=score_map,
            candidate_mask=candidate_mask,
            regions=regions,
            allocation=allocation,
        )

    def __call__(
        self,
        inputs: SelectorInput,
        budget: SelectionBudget,
    ) -> SelectionResult:
        return self.select(inputs, budget)
