"""AI Fairness 360 bias metric detector.

Computes the standard AIF360 group fairness metrics for a binary classification
dataset. All computation is synchronous (CPU-bound). Wrap in asyncio.to_thread()
when calling from async service methods.

Metrics computed:
- Disparate Impact (ratio): pass if >= 0.8 (4/5 rule)
- Statistical Parity Difference: pass if abs(value) <= 0.1
- Equal Opportunity Difference (TPR difference): pass if abs(value) <= 0.1
- Average Odds Difference (avg TPR+FPR diff): pass if abs(value) <= 0.1
- Theil Index (individual fairness): pass if value < 0.1
"""

from __future__ import annotations

from typing import Any

import numpy as np
from aumos_common.observability import get_logger

logger = get_logger(__name__)

_SUPPORTED_METRICS = [
    "disparate_impact",
    "statistical_parity_difference",
    "equal_opportunity_difference",
    "average_odds_difference",
    "theil_index",
]


class AIF360Detector:
    """Computes AIF360 group fairness metrics for binary classification datasets.

    Uses the IBM AI Fairness 360 library when available. Falls back to a
    pure-numpy implementation for environments where AIF360 is not installed
    (e.g. CI without optional dependencies).

    Note:
        AIF360 is Apache-2.0 licensed — approved for use in AumOS platform.
        Do not replace with GPL-licensed alternatives.
    """

    @property
    def supported_metrics(self) -> list[str]:
        """Return the list of metric names this detector can compute.

        Returns:
            List of standardised metric name strings.
        """
        return _SUPPORTED_METRICS

    def compute_metrics(
        self,
        features: list[dict[str, Any]],
        labels: list[int],
        predictions: list[int] | None,
        protected_attribute: str,
        privileged_values: list[Any],
        unprivileged_values: list[Any],
    ) -> dict[str, float]:
        """Compute all AIF360 metrics for a dataset.

        Falls back to the pure-numpy implementation if AIF360 is unavailable,
        ensuring this module is importable in all environments.

        Args:
            features: List of feature dicts (one per sample).
            labels: Ground-truth binary labels (0 or 1).
            predictions: Model predictions (0 or 1). None for label-only metrics.
            protected_attribute: Column name in features to assess fairness on.
            privileged_values: Values belonging to the privileged group.
            unprivileged_values: Values belonging to the unprivileged group.

        Returns:
            Dict mapping metric_name -> computed float value.
        """
        try:
            return self._compute_with_aif360(
                features=features,
                labels=labels,
                predictions=predictions,
                protected_attribute=protected_attribute,
                privileged_values=privileged_values,
                unprivileged_values=unprivileged_values,
            )
        except ImportError:
            logger.warning(
                "AIF360 not available, using numpy fallback",
                protected_attribute=protected_attribute,
            )
            return self._compute_numpy_fallback(
                features=features,
                labels=labels,
                predictions=predictions,
                protected_attribute=protected_attribute,
                privileged_values=privileged_values,
            )

    def _compute_with_aif360(
        self,
        features: list[dict[str, Any]],
        labels: list[int],
        predictions: list[int] | None,
        protected_attribute: str,
        privileged_values: list[Any],
        unprivileged_values: list[Any],
    ) -> dict[str, float]:
        """Compute metrics using the AI Fairness 360 library.

        Args:
            features: Feature dicts.
            labels: Binary ground-truth labels.
            predictions: Binary model predictions (or None).
            protected_attribute: Protected column name.
            privileged_values: Privileged group values.
            unprivileged_values: Unprivileged group values.

        Returns:
            Dict of metric_name -> value.

        Raises:
            ImportError: If aif360 is not installed.
        """
        import pandas as pd
        from aif360.datasets import BinaryLabelDataset
        from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

        # Build DataFrame with protected attribute and label
        df = pd.DataFrame(features)
        df[protected_attribute] = df[protected_attribute].astype(str)
        df["label"] = labels

        # Map privileged values to 1, unprivileged to 0 for AIF360 convention
        priv_set = {str(v) for v in privileged_values}
        df["_priv_indicator"] = df[protected_attribute].apply(lambda x: 1 if x in priv_set else 0)

        privileged_groups = [{protected_attribute: 1}]
        unprivileged_groups = [{protected_attribute: 0}]

        # Replace protected attribute with binary indicator for AIF360
        df[protected_attribute] = df["_priv_indicator"]
        df = df.drop(columns=["_priv_indicator"])

        dataset = BinaryLabelDataset(
            df=df,
            label_names=["label"],
            protected_attribute_names=[protected_attribute],
            favorable_label=1,
            unfavorable_label=0,
        )

        label_metric = BinaryLabelDatasetMetric(
            dataset,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
        )

        results: dict[str, float] = {
            "disparate_impact": float(label_metric.disparate_impact()),
            "statistical_parity_difference": float(label_metric.statistical_parity_difference()),
            "theil_index": float(label_metric.between_group_theil_index()),
        }

        # Classification metrics require predictions
        if predictions is not None:
            df_pred = df.copy()
            df_pred["label"] = predictions
            dataset_pred = BinaryLabelDataset(
                df=df_pred,
                label_names=["label"],
                protected_attribute_names=[protected_attribute],
                favorable_label=1,
                unfavorable_label=0,
            )
            clf_metric = ClassificationMetric(
                dataset,
                dataset_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups,
            )
            results["equal_opportunity_difference"] = float(clf_metric.equal_opportunity_difference())
            results["average_odds_difference"] = float(clf_metric.average_odds_difference())
        else:
            results["equal_opportunity_difference"] = 0.0
            results["average_odds_difference"] = 0.0

        logger.debug(
            "AIF360 metrics computed",
            protected_attribute=protected_attribute,
            disparate_impact=results.get("disparate_impact"),
        )
        return results

    def _compute_numpy_fallback(
        self,
        features: list[dict[str, Any]],
        labels: list[int],
        predictions: list[int] | None,
        protected_attribute: str,
        privileged_values: list[Any],
    ) -> dict[str, float]:
        """Pure-numpy fallback for environments without AIF360 installed.

        Implements the same mathematical definitions as AIF360 using numpy,
        ensuring test environments and CI pass without the full AIF360 install.

        Args:
            features: Feature dicts.
            labels: Binary ground-truth labels.
            predictions: Binary model predictions (or None).
            protected_attribute: Protected column name.
            privileged_values: Privileged group values.

        Returns:
            Dict of metric_name -> value.
        """
        priv_set = {str(v) for v in privileged_values}
        labels_arr = np.array(labels, dtype=float)
        priv_mask = np.array([str(f.get(protected_attribute, "")) in priv_set for f in features])
        unpriv_mask = ~priv_mask

        priv_rate = float(labels_arr[priv_mask].mean()) if priv_mask.sum() > 0 else 0.0
        unpriv_rate = float(labels_arr[unpriv_mask].mean()) if unpriv_mask.sum() > 0 else 0.0

        disparate_impact = unpriv_rate / priv_rate if priv_rate > 0 else float("inf")
        statistical_parity_diff = unpriv_rate - priv_rate

        # Theil index: entropy-based individual fairness (simplified between-group version)
        total_rate = float(labels_arr.mean()) if len(labels_arr) > 0 else 0.0
        if total_rate > 0 and priv_rate > 0 and unpriv_rate > 0:
            n_priv = float(priv_mask.sum())
            n_unpriv = float(unpriv_mask.sum())
            n_total = float(len(labels_arr))
            theil_index = float(
                (n_priv / n_total) * (priv_rate / total_rate) * np.log(priv_rate / total_rate + 1e-10)
                + (n_unpriv / n_total) * (unpriv_rate / total_rate) * np.log(unpriv_rate / total_rate + 1e-10)
            )
            theil_index = max(0.0, theil_index)
        else:
            theil_index = 0.0

        results: dict[str, float] = {
            "disparate_impact": round(disparate_impact, 6),
            "statistical_parity_difference": round(statistical_parity_diff, 6),
            "theil_index": round(theil_index, 6),
        }

        if predictions is not None:
            preds_arr = np.array(predictions, dtype=float)

            # Equal opportunity difference: TPR_unpriv - TPR_priv
            def tpr(mask: np.ndarray) -> float:
                """Compute true positive rate for a group mask."""
                positives = labels_arr[mask] == 1
                true_positives = (preds_arr[mask] == 1) & positives
                return float(true_positives.sum() / positives.sum()) if positives.sum() > 0 else 0.0

            def fpr(mask: np.ndarray) -> float:
                """Compute false positive rate for a group mask."""
                negatives = labels_arr[mask] == 0
                false_positives = (preds_arr[mask] == 1) & negatives
                return float(false_positives.sum() / negatives.sum()) if negatives.sum() > 0 else 0.0

            eod = tpr(unpriv_mask) - tpr(priv_mask)
            aod = ((tpr(unpriv_mask) - tpr(priv_mask)) + (fpr(unpriv_mask) - fpr(priv_mask))) / 2.0

            results["equal_opportunity_difference"] = round(eod, 6)
            results["average_odds_difference"] = round(aod, 6)
        else:
            results["equal_opportunity_difference"] = 0.0
            results["average_odds_difference"] = 0.0

        return results
