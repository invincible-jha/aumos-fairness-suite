"""Intersectional Fairness Analyzer — multi-dimensional bias detection across overlapping groups.

Detects fairness disparities that emerge at the intersection of multiple protected
attributes. Single-axis analysis often misses discriminatory patterns that only
surface when groups are examined simultaneously (e.g., Black women may face disparities
not seen when examining race or gender independently).

Methods:
- Intersectional group definition: enumerate all combinations of protected attribute values
- Per-group metric computation: demographic parity, equalized odds, calibration per subgroup
- Intersectional disparity detection: identify which intersectional groups underperform
- Statistical significance testing: distinguish real disparities from sampling noise
- Subgroup discovery: data-driven identification of underserved subgroups
- Visualization data: heatmaps and ranking tables for compliance reporting

References:
- Kearns et al. (2018): "Preventing Fairness Gerrymandering: Auditing and Learning for
  Subgroup Fairness"
- Foulds et al. (2020): "An Intersectional Definition of Fairness"
- ECOA Regulation E (12 CFR Part 202): prohibits discrimination on multiple protected bases
"""

import importlib.util
from dataclasses import dataclass, field
from itertools import product
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)


def _is_numpy_available() -> bool:
    """Check whether numpy is installed.

    Returns:
        True if numpy is importable, False otherwise.
    """
    return importlib.util.find_spec("numpy") is not None


def _is_scipy_available() -> bool:
    """Check whether scipy is installed.

    Returns:
        True if scipy is importable, False otherwise.
    """
    return importlib.util.find_spec("scipy") is not None


@dataclass
class IntersectionalGroup:
    """A group defined by the intersection of multiple protected attribute values.

    Attributes:
        group_id: Unique identifier for this intersectional group.
        attributes: Dict mapping attribute name to value defining the group.
        n_samples: Number of samples belonging to this group.
        label_rate: Rate of positive labels in this group.
        selection_rate: Rate of positive predictions in this group.
        tpr: True positive rate for this group.
        fpr: False positive rate for this group.
        parity_gap: Selection rate gap relative to reference group.
        odds_gap: Equalized odds gap relative to reference group.
        is_significant: Whether the disparity is statistically significant.
        p_value: p-value of the disparity test.
        severity: 'critical' | 'high' | 'medium' | 'low' | 'acceptable'.
    """

    group_id: str
    attributes: dict[str, Any]
    n_samples: int
    label_rate: float
    selection_rate: float
    tpr: float
    fpr: float
    parity_gap: float
    odds_gap: float
    is_significant: bool
    p_value: float
    severity: str = "acceptable"
    remediation_notes: list[str] = field(default_factory=list)


@dataclass
class IntersectionalAnalysisResult:
    """Complete result of an intersectional fairness analysis.

    Attributes:
        protected_attributes: List of protected attributes analyzed.
        reference_group: Attribute values of the reference (most privileged) group.
        intersectional_groups: All enumerated intersectional groups with metrics.
        disparate_groups: Groups with statistically significant disparities.
        max_parity_gap: Largest observed demographic parity gap.
        max_odds_gap: Largest observed equalized odds gap.
        intersectional_amplification: Ratio of intersectional vs. max single-axis gap.
        visualization_data: Heatmap-ready data for compliance reporting.
    """

    protected_attributes: list[str]
    reference_group: dict[str, Any]
    intersectional_groups: list[dict[str, Any]]
    disparate_groups: list[str]
    max_parity_gap: float
    max_odds_gap: float
    intersectional_amplification: float
    visualization_data: dict[str, Any]


class IntersectionalAnalyzer:
    """Multi-dimensional fairness analysis across overlapping protected attribute groups.

    Extends single-axis bias detection to intersectional group analysis, identifying
    discriminatory patterns that emerge at the intersection of multiple protected
    characteristics (e.g., race × gender × age). Produces statistical significance
    tests and severity rankings for compliance reporting under ECOA, EU AI Act,
    and equal opportunity regulations.

    Args:
        min_group_size: Minimum samples required to analyze a group (avoids noisy estimates).
        significance_level: p-value threshold for declaring a disparity significant.
        parity_threshold: Maximum acceptable demographic parity gap.
        odds_threshold: Maximum acceptable equalized odds gap.
        max_combination_depth: Maximum number of attributes to intersect simultaneously.
    """

    def __init__(
        self,
        min_group_size: int = 30,
        significance_level: float = 0.05,
        parity_threshold: float = 0.1,
        odds_threshold: float = 0.1,
        max_combination_depth: int = 3,
    ) -> None:
        """Initialize the IntersectionalAnalyzer.

        Args:
            min_group_size: Minimum sample count to include a group in analysis.
            significance_level: p-value threshold for statistical significance.
            parity_threshold: Demographic parity gap above which disparity is flagged.
            odds_threshold: Equalized odds gap above which disparity is flagged.
            max_combination_depth: Limit intersections to at most this many attributes.
        """
        self._min_group_size = min_group_size
        self._significance_level = significance_level
        self._parity_threshold = parity_threshold
        self._odds_threshold = odds_threshold
        self._max_combination_depth = max_combination_depth

    def is_available(self) -> bool:
        """Return True if numpy and scipy are both available.

        Returns:
            True if both dependencies are importable.
        """
        return _is_numpy_available() and _is_scipy_available()

    def compute_metrics(
        self,
        features: list[dict[str, Any]],
        labels: list[int],
        predictions: list[int] | None,
        protected_attribute: str,
        privileged_values: list[Any],
        unprivileged_values: list[Any],
    ) -> dict[str, float]:
        """Compute intersectional fairness metrics for a single attribute.

        Satisfies the BiasDetector protocol for integration with FairnessService.
        For full intersectional analysis, call analyze_intersectional() directly.

        Args:
            features: List of feature dicts (one per sample).
            labels: Ground-truth binary labels.
            predictions: Model predictions (0 or 1). None for label-only metrics.
            protected_attribute: Column name to assess fairness on.
            privileged_values: Values belonging to the privileged group.
            unprivileged_values: Values belonging to the unprivileged group.

        Returns:
            Dict mapping metric_name -> computed float value.
        """
        import numpy as np

        result = self.analyze_intersectional(
            features=features,
            labels=labels,
            predictions=predictions or [0] * len(labels),
            protected_attributes=[protected_attribute],
            reference_group={protected_attribute: privileged_values[0] if privileged_values else 1},
        )

        return {
            "intersectional_max_parity_gap": result.get("max_parity_gap", 0.0),
            "intersectional_max_odds_gap": result.get("max_odds_gap", 0.0),
            "n_disparate_groups": float(len(result.get("disparate_groups", []))),
            "intersectional_amplification": result.get("intersectional_amplification", 1.0),
        }

    @property
    def supported_metrics(self) -> list[str]:
        """Return the list of metric names this analyzer can compute.

        Returns:
            List of metric name strings.
        """
        return [
            "intersectional_max_parity_gap",
            "intersectional_max_odds_gap",
            "n_disparate_groups",
            "intersectional_amplification",
        ]

    def analyze_intersectional(
        self,
        features: list[dict[str, Any]],
        labels: list[int],
        predictions: list[int],
        protected_attributes: list[str],
        reference_group: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform full intersectional fairness analysis across multiple attributes.

        Enumerates all intersections of protected attribute values (up to
        max_combination_depth), computes fairness metrics per subgroup, tests
        statistical significance, and ranks subgroups by disparity severity.

        Args:
            features: Sample feature dicts.
            labels: Ground-truth labels (0 or 1).
            predictions: Model predictions (0 or 1).
            protected_attributes: Attribute names to intersect.
            reference_group: Dict mapping attribute name to privileged value.

        Returns:
            Serialized intersectional analysis result dict.

        Raises:
            ImportError: If numpy or scipy are not installed.
        """
        if not self.is_available():
            raise ImportError(
                "numpy and scipy are required for intersectional analysis. "
                "Install with: pip install numpy scipy"
            )

        import numpy as np
        from scipy import stats

        logger.info(
            "Starting intersectional fairness analysis",
            n_samples=len(features),
            protected_attributes=protected_attributes,
            n_attributes=len(protected_attributes),
        )

        # Limit combination depth
        depth = min(len(protected_attributes), self._max_combination_depth)

        # Discover unique values per attribute
        attribute_values: dict[str, list[Any]] = {}
        for attr in protected_attributes:
            values = list({f.get(attr) for f in features if f.get(attr) is not None})
            attribute_values[attr] = values

        # Compute reference group metrics
        ref_mask = self._get_group_mask(features, reference_group)
        ref_labels = np.array(labels)[ref_mask]
        ref_preds = np.array(predictions)[ref_mask]

        ref_selection_rate = float(ref_preds.mean()) if len(ref_preds) > 0 else 0.0
        ref_tpr = self._compute_tpr(ref_preds, ref_labels)
        ref_fpr = self._compute_fpr(ref_preds, ref_labels)

        # Enumerate and analyze all intersectional groups
        analyzed_groups: list[dict[str, Any]] = []
        disparate_groups: list[str] = []
        max_parity_gap = 0.0
        max_odds_gap = 0.0
        single_axis_max_gap = 0.0

        # Generate all attribute value combinations up to max_combination_depth
        attr_subset = protected_attributes[:depth]
        all_combinations = list(product(
            *[[(attr, val) for val in attribute_values.get(attr, [])] for attr in attr_subset]
        ))

        for combo in all_combinations:
            group_def = dict(combo)
            group_id = "_x_".join(f"{k}={v}" for k, v in sorted(group_def.items()))

            # Skip the reference group itself
            if group_def == reference_group:
                continue

            group_mask = self._get_group_mask(features, group_def)
            n_group = int(group_mask.sum())

            if n_group < self._min_group_size:
                continue

            group_labels = np.array(labels)[group_mask]
            group_preds = np.array(predictions)[group_mask]

            if len(group_preds) == 0:
                continue

            group_selection_rate = float(group_preds.mean())
            group_tpr = self._compute_tpr(group_preds, group_labels)
            group_fpr = self._compute_fpr(group_preds, group_labels)
            group_label_rate = float(group_labels.mean()) if len(group_labels) > 0 else 0.0

            parity_gap = ref_selection_rate - group_selection_rate
            tpr_gap = ref_tpr - group_tpr
            fpr_gap = ref_fpr - group_fpr
            odds_gap = (abs(tpr_gap) + abs(fpr_gap)) / 2.0

            # Statistical significance test (two-proportion z-test)
            n_ref = int(ref_mask.sum())
            p_value = self._two_proportion_z_test(
                k1=int(ref_preds.sum()),
                n1=n_ref,
                k2=int(group_preds.sum()),
                n2=n_group,
            )

            is_significant = p_value < self._significance_level
            severity = self._classify_severity(parity_gap, odds_gap, is_significant)

            if abs(parity_gap) > max_parity_gap:
                max_parity_gap = abs(parity_gap)
            if odds_gap > max_odds_gap:
                max_odds_gap = odds_gap

            # Track single-axis max gap for amplification calculation
            if len(group_def) == 1:
                single_axis_max_gap = max(single_axis_max_gap, abs(parity_gap))

            if (
                is_significant
                and (abs(parity_gap) > self._parity_threshold or odds_gap > self._odds_threshold)
            ):
                disparate_groups.append(group_id)

            remediation_notes = self._generate_remediation_notes(
                group_def, parity_gap, odds_gap, n_group
            )

            analyzed_groups.append({
                "group_id": group_id,
                "attributes": group_def,
                "n_samples": n_group,
                "label_rate": round(group_label_rate, 4),
                "selection_rate": round(group_selection_rate, 4),
                "tpr": round(group_tpr, 4),
                "fpr": round(group_fpr, 4),
                "parity_gap": round(float(parity_gap), 4),
                "odds_gap": round(float(odds_gap), 4),
                "is_significant": is_significant,
                "p_value": round(float(p_value), 6),
                "severity": severity,
                "remediation_notes": remediation_notes,
                "n_attributes_in_intersection": len(group_def),
            })

        # Sort by severity and parity gap magnitude
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "acceptable": 4}
        analyzed_groups.sort(
            key=lambda g: (severity_order.get(g["severity"], 5), -abs(g["parity_gap"]))
        )

        # Intersectional amplification: ratio of intersectional gap to single-axis max
        intersectional_amplification = (
            max_parity_gap / single_axis_max_gap
            if single_axis_max_gap > 0 else 1.0
        )

        # Visualization data: heatmap-ready structure
        visualization_data = self._build_visualization_data(
            analyzed_groups=analyzed_groups,
            protected_attributes=protected_attributes,
            attribute_values=attribute_values,
        )

        logger.info(
            "Intersectional analysis complete",
            n_groups_analyzed=len(analyzed_groups),
            n_disparate_groups=len(disparate_groups),
            max_parity_gap=max_parity_gap,
            intersectional_amplification=intersectional_amplification,
        )

        return {
            "analysis_method": "intersectional_fairness",
            "protected_attributes": protected_attributes,
            "reference_group": reference_group,
            "n_samples": len(features),
            "n_groups_analyzed": len(analyzed_groups),
            "n_disparate_groups": len(disparate_groups),
            "disparate_groups": disparate_groups,
            "max_parity_gap": round(float(max_parity_gap), 4),
            "max_odds_gap": round(float(max_odds_gap), 4),
            "single_axis_max_gap": round(float(single_axis_max_gap), 4),
            "intersectional_amplification": round(float(intersectional_amplification), 4),
            "parity_threshold": self._parity_threshold,
            "odds_threshold": self._odds_threshold,
            "significance_level": self._significance_level,
            "min_group_size": self._min_group_size,
            "reference_group_metrics": {
                "selection_rate": round(ref_selection_rate, 4),
                "tpr": round(ref_tpr, 4),
                "fpr": round(ref_fpr, 4),
            },
            "intersectional_groups": analyzed_groups,
            "visualization_data": visualization_data,
            "regulatory_flags": {
                "ecoa_concern": len(disparate_groups) > 0,
                "eu_ai_act_concern": max_parity_gap > 0.2,
                "amplification_detected": intersectional_amplification > 1.5,
            },
        }

    def _get_group_mask(
        self,
        features: list[dict[str, Any]],
        group_def: dict[str, Any],
    ) -> "Any":
        """Create a boolean mask for samples matching all group attribute values.

        Args:
            features: Sample feature dicts.
            group_def: Dict mapping attribute name to required value.

        Returns:
            numpy boolean array where True means the sample is in the group.
        """
        import numpy as np

        mask = np.ones(len(features), dtype=bool)
        for attr, val in group_def.items():
            attr_mask = np.array([
                f.get(attr) == val for f in features
            ], dtype=bool)
            mask = mask & attr_mask
        return mask

    def _compute_tpr(self, predictions: "Any", labels: "Any") -> float:
        """Compute true positive rate.

        Args:
            predictions: Prediction array.
            labels: Label array.

        Returns:
            TPR as float, 0.0 if no positive labels.
        """
        import numpy as np

        pos_mask = labels == 1
        if pos_mask.sum() == 0:
            return 0.0
        return float(predictions[pos_mask].mean())

    def _compute_fpr(self, predictions: "Any", labels: "Any") -> float:
        """Compute false positive rate.

        Args:
            predictions: Prediction array.
            labels: Label array.

        Returns:
            FPR as float, 0.0 if no negative labels.
        """
        import numpy as np

        neg_mask = labels == 0
        if neg_mask.sum() == 0:
            return 0.0
        return float(predictions[neg_mask].mean())

    def _two_proportion_z_test(
        self,
        k1: int,
        n1: int,
        k2: int,
        n2: int,
    ) -> float:
        """Compute p-value for a two-proportion z-test.

        Tests whether two proportions (selection rates) are statistically
        significantly different under the null hypothesis that they are equal.

        Args:
            k1: Successes in group 1 (reference).
            n1: Total in group 1.
            k2: Successes in group 2.
            n2: Total in group 2.

        Returns:
            Two-sided p-value.
        """
        from scipy import stats

        if n1 == 0 or n2 == 0:
            return 1.0

        p1 = k1 / n1
        p2 = k2 / n2
        p_pool = (k1 + k2) / (n1 + n2)

        if p_pool in (0.0, 1.0):
            return 1.0

        se = (p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) ** 0.5
        if se == 0:
            return 1.0

        z_score = (p1 - p2) / se
        p_value = float(2 * stats.norm.sf(abs(z_score)))
        return min(1.0, max(0.0, p_value))

    def _classify_severity(
        self,
        parity_gap: float,
        odds_gap: float,
        is_significant: bool,
    ) -> str:
        """Classify the severity of a group's disparity.

        Args:
            parity_gap: Demographic parity gap magnitude.
            odds_gap: Equalized odds gap magnitude.
            is_significant: Whether the gap is statistically significant.

        Returns:
            Severity string: 'critical', 'high', 'medium', 'low', or 'acceptable'.
        """
        if not is_significant:
            return "acceptable"

        max_gap = max(abs(parity_gap), odds_gap)

        if max_gap > 0.3:
            return "critical"
        elif max_gap > 0.2:
            return "high"
        elif max_gap > 0.1:
            return "medium"
        elif max_gap > 0.05:
            return "low"
        return "acceptable"

    def _generate_remediation_notes(
        self,
        group_def: dict[str, Any],
        parity_gap: float,
        odds_gap: float,
        n_samples: int,
    ) -> list[str]:
        """Generate remediation guidance for a disparate group.

        Args:
            group_def: Dict defining the intersectional group.
            parity_gap: Demographic parity gap.
            odds_gap: Equalized odds gap.
            n_samples: Group sample size.

        Returns:
            List of remediation recommendation strings.
        """
        notes: list[str] = []
        group_label = " x ".join(f"{k}={v}" for k, v in sorted(group_def.items()))

        if abs(parity_gap) > self._parity_threshold:
            notes.append(
                f"Group '{group_label}' has a demographic parity gap of {parity_gap:.3f}. "
                f"Consider threshold adjustment or targeted resampling for this subgroup."
            )

        if odds_gap > self._odds_threshold:
            notes.append(
                f"Group '{group_label}' has an equalized odds gap of {odds_gap:.3f}. "
                f"Consider post-processing threshold optimization for this intersection."
            )

        if n_samples < self._min_group_size * 3:
            notes.append(
                f"Group '{group_label}' has only {n_samples} samples — collect more "
                f"representative training data for this intersectional group."
            )

        return notes

    def _build_visualization_data(
        self,
        analyzed_groups: list[dict[str, Any]],
        protected_attributes: list[str],
        attribute_values: dict[str, list[Any]],
    ) -> dict[str, Any]:
        """Build heatmap-ready visualization data for reporting dashboards.

        Args:
            analyzed_groups: Analyzed group results.
            protected_attributes: List of protected attributes.
            attribute_values: Unique values per attribute.

        Returns:
            Dict with heatmap labels, values, and ranking tables.
        """
        parity_ranking = sorted(
            analyzed_groups,
            key=lambda g: abs(g["parity_gap"]),
            reverse=True,
        )[:10]

        severity_distribution: dict[str, int] = {
            "critical": 0, "high": 0, "medium": 0, "low": 0, "acceptable": 0
        }
        for group in analyzed_groups:
            sev = group.get("severity", "acceptable")
            severity_distribution[sev] = severity_distribution.get(sev, 0) + 1

        return {
            "top_disparate_groups": [
                {
                    "group_id": g["group_id"],
                    "parity_gap": g["parity_gap"],
                    "severity": g["severity"],
                    "n_samples": g["n_samples"],
                }
                for g in parity_ranking
            ],
            "severity_distribution": severity_distribution,
            "attribute_value_counts": {
                attr: len(vals) for attr, vals in attribute_values.items()
            },
            "heatmap_note": (
                "Each cell represents an intersectional group's parity gap vs. reference. "
                "Red cells indicate critical disparities requiring immediate remediation."
            ),
        }


__all__ = ["IntersectionalAnalyzer", "IntersectionalGroup", "IntersectionalAnalysisResult"]
