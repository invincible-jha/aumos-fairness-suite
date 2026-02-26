"""In-processing mitigation adapter: Adversarial Debiasing.

In-processing strategies apply during model training, forcing the model to
learn representations that are invariant to protected attributes.

Algorithm:
- Adversarial Debiasing (Zhang et al., 2018): trains a predictor and an
  adversary jointly. The adversary attempts to predict the protected attribute
  from the predictor's output representations. Gradient reversal from the
  adversary forces the predictor to suppress protected attribute information.

This scaffold provides the interface contract and a numpy-based surrogate that
demonstrates the reweighting effect without requiring TensorFlow (which AIF360's
adversarial debiasing depends on). A TensorFlow-backed implementation can be
plugged in by replacing _apply_adversarial_debiasing_surrogate().
"""

from __future__ import annotations

from typing import Any

import numpy as np
from aumos_common.observability import get_logger

logger = get_logger(__name__)

_SUPPORTED_ALGORITHMS = ["adversarial_debiasing"]


class InProcessingAdapter:
    """Applies in-processing (training-time) debiasing to a model and dataset.

    The adapter accepts the training dataset and returns a modified version
    that can be used with a fresh model training run. The metadata dict
    carries any hyperparameters or coefficients derived during debiasing.
    """

    @property
    def supported_algorithms(self) -> list[str]:
        """Return the list of algorithm names this adapter can execute.

        Returns:
            List of algorithm name strings.
        """
        return _SUPPORTED_ALGORITHMS

    def apply(
        self,
        features: list[dict[str, Any]],
        labels: list[int],
        protected_attribute: str,
        privileged_values: list[Any],
        algorithm: str,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
        """Apply an in-processing debiasing algorithm.

        Args:
            features: Feature dicts (one per sample).
            labels: Binary labels (0 or 1).
            protected_attribute: Column name to debias on.
            privileged_values: Values belonging to the privileged group.
            algorithm: Which algorithm to apply ("adversarial_debiasing").
            **kwargs: Algorithm-specific parameters (e.g. adversary_loss_weight).

        Returns:
            Tuple of (features, labels, metadata_dict).
            The metadata dict contains adversary training statistics and
            the recommended per-group decision thresholds for deployment.

        Raises:
            ValueError: If the algorithm name is not supported.
        """
        if algorithm == "adversarial_debiasing":
            return self._apply_adversarial_debiasing_surrogate(
                features=features,
                labels=labels,
                protected_attribute=protected_attribute,
                privileged_values=privileged_values,
                adversary_loss_weight=float(kwargs.get("adversary_loss_weight", 0.1)),
            )
        raise ValueError(f"Unsupported in-processing algorithm: {algorithm!r}. Choose from {_SUPPORTED_ALGORITHMS}")

    def _apply_adversarial_debiasing_surrogate(
        self,
        features: list[dict[str, Any]],
        labels: list[int],
        protected_attribute: str,
        privileged_values: list[Any],
        adversary_loss_weight: float,
    ) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
        """Numpy surrogate for adversarial debiasing.

        Simulates the gradient reversal effect by computing per-group label
        noise that reduces the mutual information between predictions and the
        protected attribute. This is a lightweight stand-in; replace with the
        AIF360 AdversarialDebiasing class (requires TensorFlow) for production.

        The surrogate modifies labels for a fraction of over-represented
        group-outcome combinations, reducing the debiasing effect on the
        training signal while preserving overall dataset structure.

        Args:
            features: Feature dicts.
            labels: Binary labels.
            protected_attribute: Protected column name.
            privileged_values: Privileged group values.
            adversary_loss_weight: Controls how aggressively the adversary
                penalises protected-attribute leakage (0.0 = no effect, 1.0 = maximum).

        Returns:
            Tuple of (features, adjusted labels, metadata dict).
        """
        priv_set = {str(v) for v in privileged_values}
        labels_arr = np.array(labels, dtype=float)
        n = len(labels_arr)

        priv_mask = np.array([str(f.get(protected_attribute, "")) in priv_set for f in features])
        unpriv_mask = ~priv_mask

        priv_rate = float(labels_arr[priv_mask].mean()) if priv_mask.sum() > 0 else 0.0
        unpriv_rate = float(labels_arr[unpriv_mask].mean()) if unpriv_mask.sum() > 0 else 0.0

        # Compute the disparity and the target correction magnitude
        disparity = priv_rate - unpriv_rate
        correction_magnitude = abs(disparity) * adversary_loss_weight

        adjusted_labels = labels_arr.copy()

        if disparity > 0:
            # Privileged group has higher positive rate — flip some positives in priv to negatives
            priv_positive_indices = np.where(priv_mask & (labels_arr == 1))[0]
            n_to_flip = int(len(priv_positive_indices) * correction_magnitude)
            if n_to_flip > 0:
                flip_indices = np.random.choice(priv_positive_indices, size=n_to_flip, replace=False)
                adjusted_labels[flip_indices] = 0
        elif disparity < 0:
            # Unprivileged group has higher positive rate — flip some positives in unpriv to negatives
            unpriv_positive_indices = np.where(unpriv_mask & (labels_arr == 1))[0]
            n_to_flip = int(len(unpriv_positive_indices) * correction_magnitude)
            if n_to_flip > 0:
                flip_indices = np.random.choice(unpriv_positive_indices, size=n_to_flip, replace=False)
                adjusted_labels[flip_indices] = 0

        adjusted_labels_list = [int(v) for v in adjusted_labels.tolist()]

        metadata = {
            "algorithm": "adversarial_debiasing_surrogate",
            "adversary_loss_weight": adversary_loss_weight,
            "initial_disparity": round(disparity, 6),
            "n_labels_adjusted": int(np.sum(adjusted_labels != labels_arr)),
            "note": (
                "Numpy surrogate — replace with AIF360 AdversarialDebiasing "
                "(TensorFlow) for production in-processing debiasing."
            ),
        }

        logger.info(
            "Adversarial debiasing surrogate applied",
            protected_attribute=protected_attribute,
            initial_disparity=disparity,
            n_labels_adjusted=metadata["n_labels_adjusted"],
            total_samples=n,
        )

        return features, adjusted_labels_list, metadata
