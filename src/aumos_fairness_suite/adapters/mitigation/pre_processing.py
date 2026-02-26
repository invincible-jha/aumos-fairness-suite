"""Pre-processing mitigation adapters: Reweighting and Rejection Sampling.

Pre-processing strategies modify the training dataset before model training
to reduce bias in the underlying data distribution.

Algorithms:
- Reweighting (Kamiran & Calders, 2012): assigns instance weights to equalise
  positive label rates across protected attribute groups without removing data.
- Rejection Sampling: resamples the dataset to balance group representation
  while preserving marginal distributions as closely as possible.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from aumos_common.observability import get_logger

logger = get_logger(__name__)

_SUPPORTED_ALGORITHMS = ["reweighting", "rejection_sampling"]


class PreProcessingAdapter:
    """Applies pre-processing debiasing to a training dataset.

    Both algorithms return the original feature set (possibly resampled) with
    a weights array or modified label set that equalises group outcomes.
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
        """Apply a pre-processing debiasing algorithm to the dataset.

        Args:
            features: Feature dicts (one per sample).
            labels: Binary labels (0 or 1).
            protected_attribute: Column name to debias on.
            privileged_values: Values belonging to the privileged group.
            algorithm: Which algorithm to apply ("reweighting" or "rejection_sampling").
            **kwargs: Unused (accepted for interface compatibility).

        Returns:
            Tuple of (transformed_features, transformed_labels, metadata_dict).
            For reweighting: metadata contains the per-instance weights array.
            For rejection_sampling: features and labels are the resampled subset.

        Raises:
            ValueError: If the algorithm name is not supported.
        """
        if algorithm == "reweighting":
            return self._reweighting(
                features=features,
                labels=labels,
                protected_attribute=protected_attribute,
                privileged_values=privileged_values,
            )
        if algorithm == "rejection_sampling":
            return self._rejection_sampling(
                features=features,
                labels=labels,
                protected_attribute=protected_attribute,
                privileged_values=privileged_values,
            )
        raise ValueError(f"Unsupported pre-processing algorithm: {algorithm!r}. Choose from {_SUPPORTED_ALGORITHMS}")

    def _reweighting(
        self,
        features: list[dict[str, Any]],
        labels: list[int],
        protected_attribute: str,
        privileged_values: list[Any],
    ) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
        """Apply Kamiran & Calders (2012) reweighting.

        Assigns per-instance weights so that within each group, positive and
        negative samples receive weights that equalise the expected positive rate.
        Dataset size is preserved; only weights change.

        Args:
            features: Feature dicts.
            labels: Binary labels.
            protected_attribute: Protected column name.
            privileged_values: Privileged group values.

        Returns:
            Tuple of (original features, original labels, {"weights": weights_list}).
        """
        priv_set = {str(v) for v in privileged_values}
        labels_arr = np.array(labels, dtype=float)
        n = len(labels_arr)

        # Group masks
        priv_mask = np.array([str(f.get(protected_attribute, "")) in priv_set for f in features])
        unpriv_mask = ~priv_mask

        # Overall positive/negative rates
        p_positive = float(labels_arr.mean())
        p_negative = 1.0 - p_positive
        p_priv = float(priv_mask.mean())
        p_unpriv = 1.0 - p_priv

        def group_positive_rate(mask: np.ndarray) -> float:
            """Compute positive label rate within a group."""
            return float(labels_arr[mask].mean()) if mask.sum() > 0 else 0.0

        p_pos_priv = group_positive_rate(priv_mask)
        p_pos_unpriv = group_positive_rate(unpriv_mask)
        p_neg_priv = 1.0 - p_pos_priv
        p_neg_unpriv = 1.0 - p_pos_unpriv

        weights = np.ones(n, dtype=float)

        for i, (feat, label) in enumerate(zip(features, labels)):
            is_priv = str(feat.get(protected_attribute, "")) in priv_set
            is_positive = label == 1

            if is_priv:
                if is_positive:
                    expected = p_positive * p_priv
                    observed = p_pos_priv * p_priv
                else:
                    expected = p_negative * p_priv
                    observed = p_neg_priv * p_priv
            else:
                if is_positive:
                    expected = p_positive * p_unpriv
                    observed = p_pos_unpriv * p_unpriv
                else:
                    expected = p_negative * p_unpriv
                    observed = p_neg_unpriv * p_unpriv

            weights[i] = expected / observed if observed > 0 else 1.0

        logger.info(
            "Reweighting applied",
            n_samples=n,
            protected_attribute=protected_attribute,
            weight_min=float(weights.min()),
            weight_max=float(weights.max()),
        )

        return features, labels, {"weights": weights.tolist()}

    def _rejection_sampling(
        self,
        features: list[dict[str, Any]],
        labels: list[int],
        protected_attribute: str,
        privileged_values: list[Any],
    ) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
        """Apply rejection sampling to equalise group representation.

        Downsamples the over-represented group/outcome combinations so that
        all four group-outcome cells (priv/unpriv x positive/negative) have
        equal representation in the output dataset. Dataset size is reduced.

        Args:
            features: Feature dicts.
            labels: Binary labels.
            protected_attribute: Protected column name.
            privileged_values: Privileged group values.

        Returns:
            Tuple of (resampled features, resampled labels, {"original_size": n}).
        """
        priv_set = {str(v) for v in privileged_values}

        # Separate into four buckets: (group, label)
        buckets: dict[tuple[str, int], list[int]] = {
            ("priv", 1): [],
            ("priv", 0): [],
            ("unpriv", 1): [],
            ("unpriv", 0): [],
        }
        for idx, (feat, label) in enumerate(zip(features, labels)):
            group = "priv" if str(feat.get(protected_attribute, "")) in priv_set else "unpriv"
            buckets[(group, label)].append(idx)

        # Target size: minimum bucket size (strict balance)
        target_size = min(len(indices) for indices in buckets.values() if indices) if buckets else 0

        if target_size == 0:
            logger.warning("Rejection sampling: one or more buckets are empty, returning original dataset")
            return features, labels, {"original_size": len(features), "resampled_size": len(features)}

        selected_indices: list[int] = []
        for indices in buckets.values():
            if indices:
                selected_indices.extend(random.sample(indices, min(target_size, len(indices))))

        random.shuffle(selected_indices)
        resampled_features = [features[i] for i in selected_indices]
        resampled_labels = [labels[i] for i in selected_indices]

        logger.info(
            "Rejection sampling applied",
            original_size=len(features),
            resampled_size=len(resampled_features),
            protected_attribute=protected_attribute,
        )

        return resampled_features, resampled_labels, {
            "original_size": len(features),
            "resampled_size": len(resampled_features),
        }
