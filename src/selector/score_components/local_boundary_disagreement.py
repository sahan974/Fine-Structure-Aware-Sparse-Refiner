"""Local boundary disagreement based on predicted-class gradients."""

from __future__ import annotations

from math import sqrt

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..types import SelectorInput
from .base import ScoreComponent


class LocalBoundaryDisagreement(ScoreComponent):
    r"""Calculate the normalized local confidence-gradient magnitude.

    Spatial gradients are calculated independently for every class-probability
    channel. At each pixel, the gradients belonging to the locally predicted
    class are selected before their magnitude is calculated::

        c*(p) = argmax_c P(c | p)
        D_local(p) = sqrt(G_x(c*(p), p)^2 + G_y(c*(p), p)^2) / sqrt(2)

    The 3x3 Sobel kernels are divided by four so each directional response is
    bounded by one for inputs in ``[0, 1]``. One-pixel replicate padding keeps
    the output at ``(H, W)`` without introducing artificial zero-valued edges.

    The inherited ``weight`` represents alpha in the composite selector score;
    it is applied by :class:`selector.selector.Selector`, not in ``compute``.
    """

    def __init__(
        self,
        *,
        weight: float = 1.0,
        validate_probabilities: bool = True,
        probability_tolerance: float = 1e-4,
    ) -> None:
        super().__init__(weight=weight)

        if probability_tolerance < 0.0:
            raise ValueError("probability_tolerance must be non-negative.")

        self.validate_probabilities = validate_probabilities
        self.probability_tolerance = float(probability_tolerance)

    def compute(self, inputs: SelectorInput) -> Tensor:
        """Return ``D_local`` with shape ``(B, H, W)``.

        ``inputs.probabilities`` must contain full-resolution, normalized
        probabilities in ``(B, K, H, W)`` layout. Backbone-specific resizing
        deliberately remains outside this score component.
        """

        probabilities = inputs.probabilities
        self._validate_tensor_contract(probabilities)

        computation_dtype = (
            torch.float32
            if probabilities.dtype in (torch.float16, torch.bfloat16)
            else probabilities.dtype
        )
        probabilities = probabilities.to(dtype=computation_dtype)

        if self.validate_probabilities:
            self._validate_probability_distribution(probabilities)

        number_of_classes = probabilities.shape[1]
        horizontal_kernel, vertical_kernel = self._sobel_kernels(probabilities)

        # Padding is explicit so convolution itself cannot fall back to zero
        # padding at the image or tile boundary.
        padded_probabilities = functional.pad(
            probabilities,
            pad=(1, 1, 1, 1),
            mode="replicate",
        )

        # A group per class applies the same kernel independently to each class
        # channel without mixing probability information between classes.
        horizontal_gradients = functional.conv2d(
            padded_probabilities,
            horizontal_kernel.repeat(number_of_classes, 1, 1, 1),
            groups=number_of_classes,
        )
        vertical_gradients = functional.conv2d(
            padded_probabilities,
            vertical_kernel.repeat(number_of_classes, 1, 1, 1),
            groups=number_of_classes,
        )

        predicted_classes = probabilities.argmax(dim=1, keepdim=True)
        selected_horizontal = horizontal_gradients.gather(
            dim=1,
            index=predicted_classes,
        ).squeeze(1)
        selected_vertical = vertical_gradients.gather(
            dim=1,
            index=predicted_classes,
        ).squeeze(1)

        disagreement = torch.sqrt(
            selected_horizontal.square() + selected_vertical.square()
        ) / sqrt(2.0)

        # The analytical range is [0, 1]. Clamp only floating-point overshoot;
        # this is not image-dependent score normalization.
        return disagreement.clamp(min=0.0, max=1.0)

    @staticmethod
    def _sobel_kernels(reference: Tensor) -> tuple[Tensor, Tensor]:
        horizontal = reference.new_tensor(
            [
                [-1.0, 0.0, 1.0],
                [-2.0, 0.0, 2.0],
                [-1.0, 0.0, 1.0],
            ]
        ).view(1, 1, 3, 3)
        vertical = reference.new_tensor(
            [
                [-1.0, -2.0, -1.0],
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 1.0],
            ]
        ).view(1, 1, 3, 3)

        return horizontal / 4.0, vertical / 4.0

    @staticmethod
    def _validate_tensor_contract(probabilities: Tensor) -> None:
        if not isinstance(probabilities, Tensor):
            raise TypeError("inputs.probabilities must be a torch.Tensor.")
        if probabilities.ndim != 4:
            raise ValueError(
                "inputs.probabilities must have shape (B, K, H, W); "
                f"received {tuple(probabilities.shape)}."
            )
        if any(size <= 0 for size in probabilities.shape):
            raise ValueError("Every probability tensor dimension must be non-empty.")
        if not probabilities.is_floating_point():
            raise TypeError("inputs.probabilities must use a floating-point dtype.")

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
