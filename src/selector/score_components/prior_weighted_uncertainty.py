"""Probability-weighted class-prior uncertainty score."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from ..types import SelectorInput
from .base import ScoreComponent


class PriorWeightedUncertainty(ScoreComponent):
    r"""Calculate the proposal's probability-weighted uncertainty score.

    For every pixel ``p`` and class ``c``::

        w_c = 1 / (sqrt(f_c) + epsilon)
        W_prior(p) = sum_c P(c | p) * w_c
        H(p) = -sum_c P(c | p) * log(P(c | p))
        U_prior(p) = W_prior(p) * H(p)

    ``f_c`` is the empirical training-set pixel frequency of class ``c``.
    Frequencies must follow the same class order as the probability channels.
    Supply a static frequency profile to the constructor for normal use. A
    profile in ``SelectorInput`` overrides it for multi-dataset evaluation.
    """

    def __init__(
        self,
        *,
        class_frequencies: Tensor | Sequence[float] | None = None,
        epsilon: float = 1e-6,
        weight: float = 1.0,
        validate_probabilities: bool = True,
        probability_tolerance: float = 1e-4,
    ) -> None:
        super().__init__(weight=weight)

        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        if probability_tolerance < 0.0:
            raise ValueError("probability_tolerance must be non-negative.")

        self.class_frequencies = class_frequencies
        self.epsilon = float(epsilon)
        self.validate_probabilities = validate_probabilities
        self.probability_tolerance = float(probability_tolerance)

    def compute(self, inputs: SelectorInput) -> Tensor:
        """Return ``U_prior`` with shape ``(B, H, W)``.

        Expected inputs:
            - ``inputs.probabilities``: normalized ``(B, K, H, W)`` tensor.
            - A normalized length-``K`` frequency vector supplied either to
              the constructor or through ``inputs.class_frequencies``.
        """

        probabilities = inputs.probabilities
        if not isinstance(probabilities, Tensor):
            raise TypeError("inputs.probabilities must be a torch.Tensor.")
        if probabilities.ndim != 4:
            raise ValueError(
                "inputs.probabilities must have shape (B, K, H, W); "
                f"received {tuple(probabilities.shape)}."
            )
        if not probabilities.is_floating_point():
            raise TypeError("inputs.probabilities must use a floating-point dtype.")

        # A per-call profile permits dataset switching. When it is absent, use
        # the static training profile supplied when the component was created.
        class_frequencies = inputs.class_frequencies
        if class_frequencies is None:
            class_frequencies = self.class_frequencies
        if class_frequencies is None:
            raise ValueError(
                "Class frequencies are required for U_prior. Supply them when "
                "constructing the component or through SelectorInput."
            )

        # Compute reductions in float32 when a backbone returns half precision.
        # This prevents avoidable loss of accuracy in logarithms and square roots.
        computation_dtype = (
            torch.float32
            if probabilities.dtype in (torch.float16, torch.bfloat16)
            else probabilities.dtype
        )
        probabilities = probabilities.to(dtype=computation_dtype)
        frequencies = torch.as_tensor(
            class_frequencies,
            dtype=computation_dtype,
            device=probabilities.device,
        )

        self._validate_frequencies(frequencies, probabilities.shape[1])
        if self.validate_probabilities:
            self._validate_probability_distribution(probabilities)

        # The epsilon placement follows the methodology exactly:
        # 1 / (sqrt(f_c) + epsilon), not 1 / sqrt(f_c + epsilon).
        inverse_root_weights = torch.reciprocal(
            torch.sqrt(frequencies) + self.epsilon
        )

        # Broadcast the K class weights over the batch and spatial dimensions.
        prior_factor = (
            probabilities * inverse_root_weights.view(1, -1, 1, 1)
        ).sum(dim=1)

        # Clamping is used only inside log. The original probabilities remain as
        # the multiplier, so a zero-probability class still contributes exactly 0.
        safe_probabilities = probabilities.clamp_min(self.epsilon)
        entropy = -(
            probabilities * torch.log(safe_probabilities)
        ).sum(dim=1)

        return prior_factor * entropy

    def _validate_frequencies(
        self,
        frequencies: Tensor,
        number_of_classes: int,
    ) -> None:
        if frequencies.ndim != 1:
            raise ValueError(
                "class_frequencies must be a one-dimensional vector."
            )
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

        tolerance = self.probability_tolerance
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
            atol=tolerance,
            rtol=0.0,
        ):
            raise ValueError(
                "inputs.probabilities must sum to 1 along the class dimension. "
                "Pass softmax probabilities rather than logits."
            )
