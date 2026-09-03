"""Spatial connected-component grouping for candidate pixels."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from scipy import ndimage
from torch import Tensor

from ..types import Region, SelectorInput
from .base import RegionGrouper


class SpatialConnectedComponentGrouper(RegionGrouper):
    """Group candidate pixels using spatial connectivity only.

    Components are labelled independently for every tile in the batch. The
    default 8-connectivity preserves diagonal links in thin structures. This
    class deliberately performs no morphology, size filtering, frequency
    comparison, gap bridging, or final budget allocation.

    Class strata are assigned after each component is formed. A component is
    labelled ``"minority"`` when the training frequency of its dominant
    predicted class is below ``minority_threshold``; otherwise it is labelled
    ``"majority"``.
    """

    def __init__(
        self,
        *,
        class_frequencies: Tensor | Sequence[float] | None = None,
        connectivity: int = 8,
        minority_threshold: float = 0.10,
        validate_probabilities: bool = True,
        probability_tolerance: float = 1e-4,
    ) -> None:
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be either 4 or 8.")

        try:
            minority_threshold = float(minority_threshold)
        except (TypeError, ValueError) as error:
            raise TypeError("minority_threshold must be a real number.") from error

        if not math.isfinite(minority_threshold):
            raise ValueError("minority_threshold must be finite.")
        if not 0.0 < minority_threshold <= 1.0:
            raise ValueError("minority_threshold must be in the interval (0, 1].")
        if probability_tolerance < 0.0:
            raise ValueError("probability_tolerance must be non-negative.")

        self.class_frequencies = class_frequencies
        self.connectivity = connectivity
        self.minority_threshold = minority_threshold
        self.validate_probabilities = validate_probabilities
        self.probability_tolerance = float(probability_tolerance)

    def group(
        self,
        candidate_mask: Tensor,
        score_map: Tensor,
        inputs: SelectorInput,
    ) -> Sequence[Region]:
        """Return spatial components in deterministic batch-major order.

        ``candidate_mask`` and ``score_map`` must have shape ``(B, H, W)``.
        Region pixel indices are flattened within their own tile, and each
        region records the corresponding ``batch_index``.
        """

        if not isinstance(inputs, SelectorInput):
            raise TypeError("inputs must be a SelectorInput.")

        probabilities = inputs.probabilities
        self._validate_tensor_contract(candidate_mask, score_map, probabilities)

        computation_dtype = (
            torch.float32
            if probabilities.dtype in (torch.float16, torch.bfloat16)
            else probabilities.dtype
        )
        probabilities = probabilities.detach().to(dtype=computation_dtype)

        class_frequencies = inputs.class_frequencies
        if class_frequencies is None:
            class_frequencies = self.class_frequencies
        if class_frequencies is None:
            raise ValueError(
                "Class frequencies are required to assign region strata. "
                "Supply them when constructing the grouper or through "
                "SelectorInput."
            )

        frequencies = torch.as_tensor(
            class_frequencies,
            dtype=computation_dtype,
            device=probabilities.device,
        )
        self._validate_frequencies(frequencies, probabilities.shape[1])
        if self.validate_probabilities:
            self._validate_probability_distribution(probabilities)

        structure = self._connectivity_structure()
        frequency_values = frequencies.detach().cpu().tolist()
        regions: list[Region] = []
        next_region_id = 0

        with torch.no_grad():
            for batch_index in range(candidate_mask.shape[0]):
                labelled, component_count = ndimage.label(
                    candidate_mask[batch_index].detach().cpu().numpy(),
                    structure=structure,
                )
                if component_count == 0:
                    continue

                tile_regions = self._build_tile_regions(
                    labelled=labelled,
                    component_count=component_count,
                    batch_index=batch_index,
                    first_region_id=next_region_id,
                    score_map=score_map[batch_index],
                    probabilities=probabilities[batch_index],
                    frequency_values=frequency_values,
                )
                regions.extend(tile_regions)
                next_region_id += component_count

        return regions

    def _build_tile_regions(
        self,
        *,
        labelled: np.ndarray,
        component_count: int,
        batch_index: int,
        first_region_id: int,
        score_map: Tensor,
        probabilities: Tensor,
        frequency_values: list[float],
    ) -> list[Region]:
        _, width = labelled.shape
        flat_labels = labelled.reshape(-1)
        candidate_positions = np.flatnonzero(flat_labels)
        component_labels = flat_labels[candidate_positions].astype(np.int64) - 1

        # Stable sorting groups component pixels while preserving ascending
        # flattened spatial order within every component.
        grouped_order = np.argsort(component_labels, kind="stable")
        grouped_positions = candidate_positions[grouped_order]
        areas = np.bincount(
            component_labels,
            minlength=component_count,
        ).astype(np.int64)
        offsets = np.concatenate(([0], np.cumsum(areas)))

        device_positions = torch.as_tensor(
            candidate_positions.copy(),
            dtype=torch.long,
            device=score_map.device,
        )
        summary_score_map = (
            score_map.float()
            if score_map.dtype in (torch.float16, torch.bfloat16)
            else score_map
        )
        candidate_scores = (
            summary_score_map.reshape(-1)
            .index_select(0, device_positions)
            .detach()
            .cpu()
            .numpy()
        )
        candidate_probabilities = (
            probabilities.reshape(probabilities.shape[0], -1)
            .index_select(1, device_positions)
            .transpose(0, 1)
            .detach()
            .cpu()
            .numpy()
        )

        score_sums = np.bincount(
            component_labels,
            weights=candidate_scores,
            minlength=component_count,
        )
        score_means = score_sums / areas
        score_maxima = np.full(component_count, -np.inf, dtype=np.float64)
        np.maximum.at(score_maxima, component_labels, candidate_scores)

        class_probability_mass = np.zeros(
            (component_count, probabilities.shape[0]),
            dtype=np.float64,
        )
        np.add.at(
            class_probability_mass,
            component_labels,
            candidate_probabilities,
        )
        dominant_classes = class_probability_mass.argmax(axis=1)
        dominant_mass = class_probability_mass[
            np.arange(component_count),
            dominant_classes,
        ]
        class_purities = dominant_mass / areas

        regions: list[Region] = []
        for component_index in range(component_count):
            start = int(offsets[component_index])
            end = int(offsets[component_index + 1])
            component_positions = grouped_positions[start:end]
            pixel_indices = torch.as_tensor(
                component_positions.copy(),
                dtype=torch.long,
                device=score_map.device,
            )

            y_coordinates = component_positions // width
            x_coordinates = component_positions % width
            bounding_box = (
                int(y_coordinates.min()),
                int(x_coordinates.min()),
                int(y_coordinates.max()) + 1,
                int(x_coordinates.max()) + 1,
            )

            dominant_class = int(dominant_classes[component_index])
            class_frequency = float(frequency_values[dominant_class])
            class_stratum = (
                "minority"
                if class_frequency < self.minority_threshold
                else "majority"
            )

            regions.append(
                Region(
                    region_id=first_region_id + component_index,
                    batch_index=batch_index,
                    pixel_indices=pixel_indices,
                    aggregate_score=float(score_sums[component_index]),
                    area=int(areas[component_index]),
                    dominant_class=dominant_class,
                    class_stratum=class_stratum,
                    metadata={
                        "mean_score": float(score_means[component_index]),
                        "max_score": float(score_maxima[component_index]),
                        "bounding_box": bounding_box,
                        "class_purity": float(class_purities[component_index]),
                        "dominant_class_frequency": class_frequency,
                    },
                )
            )

        return regions

    def _connectivity_structure(self) -> np.ndarray:
        if self.connectivity == 8:
            return np.ones((3, 3), dtype=np.uint8)
        return np.asarray(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=np.uint8,
        )

    @staticmethod
    def _validate_tensor_contract(
        candidate_mask: Tensor,
        score_map: Tensor,
        probabilities: Tensor,
    ) -> None:
        if not isinstance(candidate_mask, Tensor):
            raise TypeError("candidate_mask must be a torch.Tensor.")
        if candidate_mask.ndim != 3:
            raise ValueError(
                "candidate_mask must have shape (B, H, W); "
                f"received {tuple(candidate_mask.shape)}."
            )
        if candidate_mask.dtype != torch.bool:
            raise TypeError("candidate_mask must use the boolean dtype.")
        if any(size <= 0 for size in candidate_mask.shape):
            raise ValueError("Every candidate_mask dimension must be non-empty.")

        if not isinstance(score_map, Tensor):
            raise TypeError("score_map must be a torch.Tensor.")
        if score_map.shape != candidate_mask.shape:
            raise ValueError(
                "score_map and candidate_mask must have the same (B, H, W) "
                "shape."
            )
        if not score_map.is_floating_point():
            raise TypeError("score_map must use a floating-point dtype.")
        if not bool(torch.isfinite(score_map).all()):
            raise ValueError("score_map must contain only finite values.")

        if not isinstance(probabilities, Tensor):
            raise TypeError("inputs.probabilities must be a torch.Tensor.")
        if probabilities.ndim != 4:
            raise ValueError(
                "inputs.probabilities must have shape (B, K, H, W); "
                f"received {tuple(probabilities.shape)}."
            )
        expected_probability_shape = (
            candidate_mask.shape[0],
            probabilities.shape[1],
            candidate_mask.shape[1],
            candidate_mask.shape[2],
        )
        if probabilities.shape != expected_probability_shape:
            raise ValueError(
                "The probability batch and spatial dimensions must match "
                "candidate_mask."
            )
        if probabilities.shape[1] <= 0:
            raise ValueError("The probability class dimension must be non-empty.")
        if not probabilities.is_floating_point():
            raise TypeError("inputs.probabilities must use a floating-point dtype.")

        devices = {candidate_mask.device, score_map.device, probabilities.device}
        if len(devices) != 1:
            raise ValueError(
                "candidate_mask, score_map, and inputs.probabilities must be "
                "on the same device."
            )

    def _validate_frequencies(
        self,
        frequencies: Tensor,
        number_of_classes: int,
    ) -> None:
        if frequencies.ndim != 1:
            raise ValueError("class_frequencies must be one-dimensional.")
        if frequencies.numel() != number_of_classes:
            raise ValueError(
                "The number of class frequencies must match the probability "
                f"channels: expected {number_of_classes}, received "
                f"{frequencies.numel()}."
            )
        if not bool(torch.isfinite(frequencies).all()):
            raise ValueError("class_frequencies must contain finite values.")
        if bool((frequencies <= 0).any()):
            raise ValueError("Every class frequency must be greater than zero.")
        if not torch.isclose(
            frequencies.sum(),
            frequencies.new_tensor(1.0),
            atol=self.probability_tolerance,
            rtol=0.0,
        ):
            raise ValueError("class_frequencies must sum to 1.")

    def _validate_probability_distribution(self, probabilities: Tensor) -> None:
        if not bool(torch.isfinite(probabilities).all()):
            raise ValueError("inputs.probabilities must contain finite values.")
        if bool((probabilities < 0.0).any()) or bool(
            (probabilities > 1.0).any()
        ):
            raise ValueError(
                "inputs.probabilities must contain values in [0, 1]. "
                "Pass softmax probabilities rather than logits."
            )

        probability_sums = probabilities.sum(dim=1)
        if not torch.allclose(
            probability_sums,
            torch.ones_like(probability_sums),
            atol=self.probability_tolerance,
            rtol=0.0,
        ):
            raise ValueError(
                "inputs.probabilities must sum to 1 along the class dimension. "
                "Pass softmax probabilities rather than logits."
            )
