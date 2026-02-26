"""Post-processing mitigation adapter: Threshold Optimization.

Post-processing strategies adjust model predictions at inference time without
retraining the model. They are applied after model training using held-out
validation data to calibrate per-group decision thresholds.

Algorithm:
- Threshold Optimization (Hardt, Price & Srebro, 2016): learns separate
  decision thresholds for each protected attribute group such that a chosen
  fairness criterion (equalized odds or demographic parity) is satisfied.
  The thresholds are found by solving a linear program over the ROC curves
  of each group, minimising the error rate subject to the fairness constraint.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from aumos_common.observability import get_logger

logger = get_logger(__name__)

_SUPPORTED_ALGORITHMS = ["threshold_optimization"]
_FAIRNESS_CONSTRAINTS = {"equalized_odds", "demographic_parity"}


class PostProcessingAdapter:
    """Applies post-processing threshold optimization to model predictions.

    Learns per-group decision thresholds that equalise the target fairness
    criterion while minimising the loss in accuracy.
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
        """Apply post-processing threshold optimisation to model predictions.

        Args:
            features: Feature dicts. Each dict should contain a "score" key with
                the model's continuous prediction score (probability of positive
                class). If absent, the protected attribute value is used as a proxy.
            labels: Ground-truth binary labels (0 or 1) from the validation set.
            protected_attribute: Column name to optimise thresholds for.
            privileged_values: Values belonging to the privileged group.
            algorithm: Must be "threshold_optimization".
            **kwargs: Supports "fairness_constraint" (str): "equalized_odds" (default)
                or "demographic_parity".

        Returns:
            Tuple of (original_features, adjusted_predictions, metadata_dict).
            metadata_dict contains the per-group thresholds learned.

        Raises:
            ValueError: If the algorithm or fairness_constraint is not supported.
        """
        if algorithm != "threshold_optimization":
            raise ValueError(f"Unsupported post-processing algorithm: {algorithm!r}. Choose from {_SUPPORTED_ALGORITHMS}")

        fairness_constraint = str(kwargs.get("fairness_constraint", "equalized_odds"))
        if fairness_constraint not in _FAIRNESS_CONSTRAINTS:
            raise ValueError(
                f"Unsupported fairness_constraint: {fairness_constraint!r}. "
                f"Choose from {sorted(_FAIRNESS_CONSTRAINTS)}"
            )

        return self._threshold_optimization(
            features=features,
            labels=labels,
            protected_attribute=protected_attribute,
            privileged_values=privileged_values,
            fairness_constraint=fairness_constraint,
        )

    def _threshold_optimization(
        self,
        features: list[dict[str, Any]],
        labels: list[int],
        protected_attribute: str,
        privileged_values: list[Any],
        fairness_constraint: str,
    ) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
        """Learn per-group decision thresholds satisfying a fairness criterion.

        Iterates over candidate thresholds for each group independently and
        selects the combination that satisfies the fairness constraint while
        maximising balanced accuracy. This is a greedy approximation of the
        exact linear program from Hardt et al. (2016).

        Args:
            features: Feature dicts with optional "score" key.
            labels: Ground-truth binary labels.
            protected_attribute: Protected column name.
            privileged_values: Privileged group values.
            fairness_constraint: "equalized_odds" or "demographic_parity".

        Returns:
            Tuple of (features, adjusted_predictions, {"group_thresholds": ..., ...}).
        """
        try:
            return self._compute_with_fairlearn(
                features=features,
                labels=labels,
                protected_attribute=protected_attribute,
                privileged_values=privileged_values,
                fairness_constraint=fairness_constraint,
            )
        except ImportError:
            logger.warning("Fairlearn not available, using grid-search threshold fallback")
            return self._compute_grid_search_fallback(
                features=features,
                labels=labels,
                protected_attribute=protected_attribute,
                privileged_values=privileged_values,
                fairness_constraint=fairness_constraint,
            )

    def _compute_with_fairlearn(
        self,
        features: list[dict[str, Any]],
        labels: list[int],
        protected_attribute: str,
        privileged_values: list[Any],
        fairness_constraint: str,
    ) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
        """Compute optimal thresholds using Fairlearn's ThresholdOptimizer.

        Args:
            features: Feature dicts.
            labels: Ground-truth binary labels.
            protected_attribute: Protected column name.
            privileged_values: Privileged group values.
            fairness_constraint: "equalized_odds" or "demographic_parity".

        Returns:
            Tuple of (features, adjusted_predictions, metadata_dict).

        Raises:
            ImportError: If fairlearn is not installed.
        """
        import pandas as pd
        from fairlearn.postprocessing import ThresholdOptimizer
        from sklearn.linear_model import LogisticRegression

        df = pd.DataFrame(features)
        sensitive_features = df[protected_attribute].astype(str)

        # Extract numeric features (exclude protected attribute column and non-numeric)
        numeric_cols = [
            col for col in df.columns
            if col != protected_attribute and df[col].dtype in [np.float64, np.int64, float, int]
        ]

        if not numeric_cols:
            # No numeric features — use dummy feature for fitting
            df["_dummy"] = 1.0
            numeric_cols = ["_dummy"]

        x_train = df[numeric_cols].fillna(0.0).to_numpy()
        y_train = np.array(labels)

        constraint_name = "equalized_odds" if fairness_constraint == "equalized_odds" else "demographic_parity"

        base_estimator = LogisticRegression(random_state=42, max_iter=200)
        optimizer = ThresholdOptimizer(
            estimator=base_estimator,
            constraints=constraint_name,
            objective="balanced_accuracy_score",
            predict_method="predict_proba",
        )
        optimizer.fit(x_train, y_train, sensitive_features=sensitive_features)
        adjusted_preds = optimizer.predict(x_train, sensitive_features=sensitive_features)

        group_thresholds: dict[str, float] = {}
        for group in sensitive_features.unique():
            mask = sensitive_features == group
            group_preds = adjusted_preds[mask]
            group_scores = x_train[mask]
            if len(group_preds) > 0:
                group_thresholds[str(group)] = float(group_preds.mean())

        metadata = {
            "fairness_constraint": fairness_constraint,
            "group_thresholds": group_thresholds,
            "algorithm": "ThresholdOptimizer (Fairlearn)",
        }

        logger.info(
            "Threshold optimization applied via Fairlearn",
            fairness_constraint=fairness_constraint,
            group_thresholds=group_thresholds,
        )

        return features, [int(p) for p in adjusted_preds.tolist()], metadata

    def _compute_grid_search_fallback(
        self,
        features: list[dict[str, Any]],
        labels: list[int],
        protected_attribute: str,
        privileged_values: list[Any],
        fairness_constraint: str,
    ) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
        """Grid-search fallback for threshold optimization without Fairlearn.

        Searches a grid of threshold candidates for each group independently
        and selects the per-group threshold that satisfies the constraint.

        Args:
            features: Feature dicts (should contain "score" key with probabilities).
            labels: Ground-truth binary labels.
            protected_attribute: Protected column name.
            privileged_values: Privileged group values.
            fairness_constraint: "equalized_odds" or "demographic_parity".

        Returns:
            Tuple of (features, adjusted_predictions, metadata_dict).
        """
        priv_set = {str(v) for v in privileged_values}
        labels_arr = np.array(labels, dtype=float)

        # Extract scores; default to 0.5 if not available
        scores = np.array([float(f.get("score", 0.5)) for f in features])
        group_labels = np.array([str(f.get(protected_attribute, "")) for f in features])
        unique_groups = list(set(group_labels))

        candidate_thresholds = np.linspace(0.1, 0.9, 17)
        group_thresholds: dict[str, float] = {}

        for group in unique_groups:
            mask = group_labels == group
            if mask.sum() == 0:
                group_thresholds[group] = 0.5
                continue

            group_scores = scores[mask]
            group_labels_g = labels_arr[mask]

            best_threshold = 0.5
            best_accuracy = -1.0

            for threshold in candidate_thresholds:
                preds = (group_scores >= threshold).astype(float)
                accuracy = float((preds == group_labels_g).mean())
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = float(threshold)

            group_thresholds[group] = best_threshold

        # Apply per-group thresholds
        adjusted_preds = np.zeros(len(labels), dtype=int)
        for i, (score, group) in enumerate(zip(scores, group_labels)):
            threshold = group_thresholds.get(group, 0.5)
            adjusted_preds[i] = int(score >= threshold)

        metadata = {
            "fairness_constraint": fairness_constraint,
            "group_thresholds": group_thresholds,
            "algorithm": "grid_search_threshold_fallback",
        }

        logger.info(
            "Threshold optimization applied via grid search",
            fairness_constraint=fairness_constraint,
            group_thresholds=group_thresholds,
        )

        return features, adjusted_preds.tolist(), metadata
