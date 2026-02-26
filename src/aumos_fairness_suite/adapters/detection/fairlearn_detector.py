"""Microsoft Fairlearn bias metric detector.

Computes Fairlearn group fairness metrics for a binary classification dataset.
All computation is synchronous (CPU-bound). Wrap in asyncio.to_thread() when
calling from async service methods.

Metrics computed:
- Demographic Parity Difference: max selection rate difference across groups
- Demographic Parity Ratio: min selection rate ratio across groups
- Equalized Odds Difference: max of TPR and FPR differences across groups
- Equalized Odds Ratio: min of TPR and FPR ratios across groups
"""

from __future__ import annotations

from typing import Any

import numpy as np
from aumos_common.observability import get_logger

logger = get_logger(__name__)

_SUPPORTED_METRICS = [
    "demographic_parity_difference",
    "demographic_parity_ratio",
    "equalized_odds_difference",
    "equalized_odds_ratio",
]


class FairlearnDetector:
    """Computes Fairlearn group fairness metrics for binary classification datasets.

    Uses the Microsoft Fairlearn library when available. Falls back to a
    pure-numpy implementation for environments where fairlearn is not installed.

    Note:
        Fairlearn is MIT licensed — approved for use in AumOS platform.
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
        """Compute all Fairlearn metrics for a dataset.

        Falls back to the pure-numpy implementation if fairlearn is unavailable,
        ensuring this module is importable in all environments.

        Args:
            features: List of feature dicts (one per sample).
            labels: Ground-truth binary labels (0 or 1).
            predictions: Model predictions (0 or 1). None for label-only metrics.
            protected_attribute: Column name in features to assess fairness on.
            privileged_values: Values belonging to the privileged group (unused in
                Fairlearn — all groups are assessed simultaneously).
            unprivileged_values: Values belonging to the unprivileged group (unused).

        Returns:
            Dict mapping metric_name -> computed float value.
        """
        try:
            return self._compute_with_fairlearn(
                features=features,
                labels=labels,
                predictions=predictions,
                protected_attribute=protected_attribute,
            )
        except ImportError:
            logger.warning(
                "Fairlearn not available, using numpy fallback",
                protected_attribute=protected_attribute,
            )
            return self._compute_numpy_fallback(
                features=features,
                labels=labels,
                predictions=predictions,
                protected_attribute=protected_attribute,
            )

    def _compute_with_fairlearn(
        self,
        features: list[dict[str, Any]],
        labels: list[int],
        predictions: list[int] | None,
        protected_attribute: str,
    ) -> dict[str, float]:
        """Compute metrics using the Microsoft Fairlearn library.

        Args:
            features: Feature dicts.
            labels: Binary ground-truth labels.
            predictions: Binary model predictions (or None for selection-rate only).
            protected_attribute: Protected column name.

        Returns:
            Dict of metric_name -> value.

        Raises:
            ImportError: If fairlearn is not installed.
        """
        import pandas as pd
        from fairlearn.metrics import (
            MetricFrame,
            demographic_parity_difference,
            demographic_parity_ratio,
            equalized_odds_difference,
            equalized_odds_ratio,
        )

        df = pd.DataFrame(features)
        sensitive_features = df[protected_attribute].astype(str)
        labels_arr = np.array(labels)

        results: dict[str, float] = {}

        # Selection-rate metrics (require only labels, not predictions)
        # Use labels as predictions for dataset-level assessment when no model predictions
        preds_for_selection = np.array(predictions) if predictions is not None else labels_arr

        results["demographic_parity_difference"] = float(
            demographic_parity_difference(
                y_true=labels_arr,
                y_pred=preds_for_selection,
                sensitive_features=sensitive_features,
            )
        )
        results["demographic_parity_ratio"] = float(
            demographic_parity_ratio(
                y_true=labels_arr,
                y_pred=preds_for_selection,
                sensitive_features=sensitive_features,
            )
        )

        if predictions is not None:
            preds_arr = np.array(predictions)
            results["equalized_odds_difference"] = float(
                equalized_odds_difference(
                    y_true=labels_arr,
                    y_pred=preds_arr,
                    sensitive_features=sensitive_features,
                )
            )
            results["equalized_odds_ratio"] = float(
                equalized_odds_ratio(
                    y_true=labels_arr,
                    y_pred=preds_arr,
                    sensitive_features=sensitive_features,
                )
            )
        else:
            results["equalized_odds_difference"] = 0.0
            results["equalized_odds_ratio"] = 1.0

        logger.debug(
            "Fairlearn metrics computed",
            protected_attribute=protected_attribute,
            demographic_parity_difference=results.get("demographic_parity_difference"),
        )
        return results

    def _compute_numpy_fallback(
        self,
        features: list[dict[str, Any]],
        labels: list[int],
        predictions: list[int] | None,
        protected_attribute: str,
    ) -> dict[str, float]:
        """Pure-numpy fallback for environments without Fairlearn installed.

        Computes multi-group metrics by iterating over all unique group values,
        matching Fairlearn's "max difference / min ratio" semantics.

        Args:
            features: Feature dicts.
            labels: Binary ground-truth labels.
            predictions: Binary model predictions (or None).
            protected_attribute: Protected column name.

        Returns:
            Dict of metric_name -> value.
        """
        labels_arr = np.array(labels, dtype=float)
        group_values = list({str(f.get(protected_attribute, "")) for f in features})
        preds_arr = np.array(predictions, dtype=float) if predictions is not None else labels_arr

        group_selection_rates: dict[str, float] = {}
        group_tprs: dict[str, float] = {}
        group_fprs: dict[str, float] = {}

        for group in group_values:
            mask = np.array([str(f.get(protected_attribute, "")) == group for f in features])
            if mask.sum() == 0:
                continue

            group_selection_rates[group] = float(preds_arr[mask].mean())

            if predictions is not None:
                positives = labels_arr[mask] == 1
                negatives = labels_arr[mask] == 0
                tp = ((preds_arr[mask] == 1) & positives).sum()
                fp = ((preds_arr[mask] == 1) & negatives).sum()
                group_tprs[group] = float(tp / positives.sum()) if positives.sum() > 0 else 0.0
                group_fprs[group] = float(fp / negatives.sum()) if negatives.sum() > 0 else 0.0

        rates = list(group_selection_rates.values())
        if not rates:
            return {
                "demographic_parity_difference": 0.0,
                "demographic_parity_ratio": 1.0,
                "equalized_odds_difference": 0.0,
                "equalized_odds_ratio": 1.0,
            }

        dpd = max(rates) - min(rates)
        dpr = min(rates) / max(rates) if max(rates) > 0 else 0.0

        if predictions is not None and group_tprs:
            tprs = list(group_tprs.values())
            fprs = list(group_fprs.values())
            eod = max(
                max(tprs) - min(tprs),
                max(fprs) - min(fprs),
            )
            tpr_ratio = min(tprs) / max(tprs) if max(tprs) > 0 else 0.0
            fpr_ratio = min(fprs) / max(fprs) if max(fprs) > 0 else 1.0
            eor = min(tpr_ratio, fpr_ratio)
        else:
            eod = 0.0
            eor = 1.0

        return {
            "demographic_parity_difference": round(dpd, 6),
            "demographic_parity_ratio": round(dpr, 6),
            "equalized_odds_difference": round(eod, 6),
            "equalized_odds_ratio": round(eor, 6),
        }
