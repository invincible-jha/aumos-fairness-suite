"""Fairness Dashboard Service — structured data for interactive fairness visualizations.

Transforms stored fairness assessment results (fai_assessments, fai_bias_metrics)
into structured JSON payloads compatible with Recharts, Plotly, and Vega-Lite for
interactive display in the AumOS governance dashboard.

This service is a pure data transformation layer — it does NOT render charts.
It produces the precise data structures that frontend chart libraries expect:

- Model dashboard: aggregated metric summary + group-level breakdown
- Per-group analysis: metric values per protected group with pass/fail status
- Temporal history: time-series of metric results for trend visualization
- Heatmap: protected groups × metrics grid of metric scores
- Model comparison: side-by-side metric comparison between two models

Dashboard endpoints:
  GET /api/v1/fairness/dashboard/{model_id}           — model summary
  GET /api/v1/fairness/dashboard/{model_id}/groups    — per-group breakdown
  GET /api/v1/fairness/dashboard/{model_id}/history   — time-series
  GET /api/v1/fairness/dashboard/{model_id}/heatmap   — metric × group heatmap
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)


@dataclass
class MetricSummary:
    """Summary of a single fairness metric across all groups.

    Attributes:
        metric_name: Name of the fairness metric.
        value: Computed metric value.
        threshold: Pass/fail threshold for this metric.
        passed: Whether the metric passed its threshold.
        worst_group: Protected group with the worst metric value.
        best_group: Protected group with the best metric value.
    """

    metric_name: str
    value: float
    threshold: float
    passed: bool
    worst_group: str | None = None
    best_group: str | None = None


@dataclass
class GroupMetricRow:
    """Per-group metric values for a single assessment.

    Attributes:
        group_label: Human-readable group label (e.g., 'gender=female').
        group_attributes: Dict of protected attributes defining this group.
        n_samples: Number of samples in this group.
        metrics: Dict mapping metric name to (value, passed) pairs.
        overall_passed: True if all metrics passed for this group.
        risk_level: 'high', 'medium', or 'low' based on number of failed metrics.
    """

    group_label: str
    group_attributes: dict[str, Any]
    n_samples: int
    metrics: dict[str, dict[str, Any]]
    overall_passed: bool
    risk_level: str


@dataclass
class FairnessDashboard:
    """Complete dashboard data for a model's fairness assessment.

    Attributes:
        model_id: UUID of the model.
        assessment_id: UUID of the latest assessment.
        overall_passed: True if all metrics passed.
        pass_rate: Fraction of metrics that passed.
        metric_summaries: Per-metric summary list.
        group_rows: Per-group metric breakdown.
        assessment_timestamp: ISO-8601 timestamp of assessment.
        protected_attributes: List of protected attributes assessed.
    """

    model_id: str
    assessment_id: str
    overall_passed: bool
    pass_rate: float
    metric_summaries: list[MetricSummary]
    group_rows: list[GroupMetricRow]
    assessment_timestamp: str
    protected_attributes: list[str]


@dataclass
class TemporalDataPoint:
    """A single metric data point in a time series.

    Attributes:
        timestamp: ISO-8601 assessment timestamp.
        assessment_id: UUID of the assessment.
        metric_name: Name of the metric.
        value: Metric value at this point in time.
        passed: Whether the metric passed its threshold.
    """

    timestamp: str
    assessment_id: str
    metric_name: str
    value: float
    passed: bool


@dataclass
class HeatmapCell:
    """Single cell in a group × metric fairness heatmap.

    Attributes:
        group_label: Protected group label (row).
        metric_name: Metric name (column).
        value: Metric value for this group-metric combination.
        passed: Whether this cell's metric passed its threshold.
        normalized_value: Value scaled to [0, 1] for color encoding.
    """

    group_label: str
    metric_name: str
    value: float
    passed: bool
    normalized_value: float


class FairnessDashboardService:
    """Transforms fairness assessment data into dashboard-ready visualization payloads.

    Reads from pre-computed assessment results and builds structured JSON
    that frontend chart libraries (Recharts, Plotly, Vega-Lite) consume directly.
    No ML computation is performed here — all metrics are already stored.

    Args:
        default_metrics: Default list of metric names to include in dashboards.
            If None, all available metrics are included.
    """

    # Default fairness metric thresholds (configurable per tenant)
    _DEFAULT_THRESHOLDS: dict[str, float] = {
        "disparate_impact": 0.8,
        "statistical_parity_difference": 0.1,
        "equal_opportunity_difference": 0.1,
        "average_odds_difference": 0.1,
        "demographic_parity_difference": 0.1,
        "demographic_parity_ratio": 0.8,
        "equalized_odds_difference": 0.1,
        "equalized_odds_ratio": 0.8,
        "theil_index": 0.1,
    }

    def __init__(
        self,
        default_metrics: list[str] | None = None,
    ) -> None:
        """Initialise the dashboard service.

        Args:
            default_metrics: Metric names to include (None = all available).
        """
        self._default_metrics = default_metrics

    def build_model_dashboard(
        self,
        assessment: dict[str, Any],
        bias_metrics: list[dict[str, Any]],
        thresholds: dict[str, float] | None = None,
    ) -> FairnessDashboard:
        """Build a complete dashboard payload for a model's latest assessment.

        Args:
            assessment: FairnessAssessment record dict (from fai_assessments).
            bias_metrics: List of BiasMetric record dicts for this assessment.
            thresholds: Custom metric thresholds (default: _DEFAULT_THRESHOLDS).

        Returns:
            FairnessDashboard with all components populated.
        """
        thresholds = thresholds or self._DEFAULT_THRESHOLDS

        # Group bias metrics by metric name for summary computation
        by_metric: dict[str, list[dict[str, Any]]] = {}
        for metric in bias_metrics:
            metric_name = metric.get("metric_name", "unknown")
            by_metric.setdefault(metric_name, []).append(metric)

        metric_summaries: list[MetricSummary] = []
        for metric_name, metric_records in by_metric.items():
            threshold = thresholds.get(metric_name, 0.1)
            values = [float(m.get("value", 0.0)) for m in metric_records]
            overall_value = sum(values) / len(values) if values else 0.0

            # For ratio metrics, pass if >= threshold; for difference metrics, pass if <= threshold
            is_ratio_metric = "ratio" in metric_name or metric_name == "disparate_impact"
            passed = overall_value >= threshold if is_ratio_metric else abs(overall_value) <= threshold

            # Find worst/best performing groups
            groups = [m.get("group_label", "unknown") for m in metric_records]
            worst_group = None
            best_group = None
            if values and groups:
                worst_idx = values.index(min(values)) if is_ratio_metric else values.index(max(values, key=abs))
                best_idx = values.index(max(values)) if is_ratio_metric else values.index(min(values, key=abs))
                worst_group = groups[worst_idx] if worst_idx < len(groups) else None
                best_group = groups[best_idx] if best_idx < len(groups) else None

            metric_summaries.append(MetricSummary(
                metric_name=metric_name,
                value=round(overall_value, 6),
                threshold=threshold,
                passed=passed,
                worst_group=worst_group,
                best_group=best_group,
            ))

        # Build group rows
        group_rows = self._build_group_rows(bias_metrics, thresholds)

        passed_count = sum(1 for ms in metric_summaries if ms.passed)
        pass_rate = passed_count / len(metric_summaries) if metric_summaries else 1.0
        overall_passed = all(ms.passed for ms in metric_summaries)

        protected_attributes = list({
            attr
            for m in bias_metrics
            for attr in (m.get("protected_attribute", "").split(",") if m.get("protected_attribute") else [])
        })

        logger.info(
            "fairness_dashboard_built",
            model_id=str(assessment.get("model_id", "")),
            assessment_id=str(assessment.get("id", "")),
            overall_passed=overall_passed,
            pass_rate=round(pass_rate, 3),
            n_metrics=len(metric_summaries),
        )

        return FairnessDashboard(
            model_id=str(assessment.get("model_id", "")),
            assessment_id=str(assessment.get("id", "")),
            overall_passed=overall_passed,
            pass_rate=round(pass_rate, 4),
            metric_summaries=metric_summaries,
            group_rows=group_rows,
            assessment_timestamp=str(assessment.get("created_at", "")),
            protected_attributes=protected_attributes,
        )

    def build_temporal_history(
        self,
        assessments: list[dict[str, Any]],
        bias_metrics_by_assessment: dict[str, list[dict[str, Any]]],
        metric_names: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Build time-series data for metric trend visualization.

        Args:
            assessments: List of FairnessAssessment records sorted by created_at.
            bias_metrics_by_assessment: Dict mapping assessment_id to its BiasMetrics.
            metric_names: Specific metrics to include (None = all).

        Returns:
            List of data points keyed by timestamp, suitable for a line chart.
        """
        data_points: list[dict[str, Any]] = []

        for assessment in assessments:
            assessment_id = str(assessment.get("id", ""))
            timestamp = str(assessment.get("created_at", ""))
            metrics = bias_metrics_by_assessment.get(assessment_id, [])

            metric_values: dict[str, float] = {}
            metric_passed: dict[str, bool] = {}

            for metric in metrics:
                metric_name = metric.get("metric_name", "unknown")
                if metric_names and metric_name not in metric_names:
                    continue

                value = float(metric.get("value", 0.0))
                passed = bool(metric.get("passed", True))

                if metric_name not in metric_values:
                    metric_values[metric_name] = value
                    metric_passed[metric_name] = passed

            data_points.append({
                "timestamp": timestamp,
                "assessment_id": assessment_id,
                "metrics": {
                    name: {
                        "value": round(value, 6),
                        "passed": metric_passed.get(name, True),
                    }
                    for name, value in metric_values.items()
                },
            })

        return data_points

    def build_heatmap(
        self,
        bias_metrics: list[dict[str, Any]],
        thresholds: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Build a group × metric heatmap payload.

        Args:
            bias_metrics: BiasMetric records for a single assessment.
            thresholds: Metric thresholds for pass/fail determination.

        Returns:
            Dict with 'groups', 'metrics', 'cells', and 'summary' for heatmap rendering.
        """
        thresholds = thresholds or self._DEFAULT_THRESHOLDS

        groups: set[str] = set()
        metrics: set[str] = set()

        for m in bias_metrics:
            group = m.get("group_label") or m.get("protected_attribute", "all")
            groups.add(group)
            metrics.add(m.get("metric_name", "unknown"))

        sorted_groups = sorted(groups)
        sorted_metrics = sorted(metrics)

        # Build cell lookup: (group, metric) -> value
        cell_map: dict[tuple[str, str], float] = {}
        for m in bias_metrics:
            group = m.get("group_label") or m.get("protected_attribute", "all")
            metric_name = m.get("metric_name", "unknown")
            cell_map[(group, metric_name)] = float(m.get("value", 0.0))

        # Compute value range per metric for normalization
        cells: list[dict[str, Any]] = []
        for metric_name in sorted_metrics:
            metric_values = [
                cell_map.get((g, metric_name), 0.0) for g in sorted_groups
            ]
            val_min = min(metric_values) if metric_values else 0.0
            val_max = max(metric_values) if metric_values else 1.0
            val_range = val_max - val_min if val_max != val_min else 1.0

            threshold = thresholds.get(metric_name, 0.1)
            is_ratio_metric = "ratio" in metric_name or metric_name == "disparate_impact"

            for group_label in sorted_groups:
                value = cell_map.get((group_label, metric_name), 0.0)
                passed = value >= threshold if is_ratio_metric else abs(value) <= threshold
                normalized = (value - val_min) / val_range

                cells.append({
                    "group_label": group_label,
                    "metric_name": metric_name,
                    "value": round(value, 6),
                    "passed": passed,
                    "normalized_value": round(normalized, 4),
                })

        return {
            "groups": sorted_groups,
            "metrics": sorted_metrics,
            "cells": cells,
            "color_scale": "RdYlGn",
            "n_groups": len(sorted_groups),
            "n_metrics": len(sorted_metrics),
        }

    def build_model_comparison(
        self,
        model_a_assessment: dict[str, Any],
        model_a_metrics: list[dict[str, Any]],
        model_b_assessment: dict[str, Any],
        model_b_metrics: list[dict[str, Any]],
        thresholds: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Build side-by-side model comparison data.

        Args:
            model_a_assessment: Assessment record for model A.
            model_a_metrics: Bias metric records for model A.
            model_b_assessment: Assessment record for model B.
            model_b_metrics: Bias metric records for model B.
            thresholds: Metric thresholds for pass/fail determination.

        Returns:
            Dict with 'model_a', 'model_b', and 'comparison' sections.
        """
        thresholds = thresholds or self._DEFAULT_THRESHOLDS

        def _aggregate_metrics(metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
            by_metric: dict[str, list[float]] = {}
            for m in metrics:
                by_metric.setdefault(m.get("metric_name", "unknown"), []).append(
                    float(m.get("value", 0.0))
                )
            return {
                name: {
                    "mean_value": round(sum(vals) / len(vals), 6),
                    "threshold": thresholds.get(name, 0.1),
                }
                for name, vals in by_metric.items()
            }

        metrics_a = _aggregate_metrics(model_a_metrics)
        metrics_b = _aggregate_metrics(model_b_metrics)

        # Compute per-metric delta: B - A
        all_metrics = sorted(set(metrics_a) | set(metrics_b))
        comparison = []
        for metric_name in all_metrics:
            val_a = metrics_a.get(metric_name, {}).get("mean_value", 0.0)
            val_b = metrics_b.get(metric_name, {}).get("mean_value", 0.0)
            delta = val_b - val_a
            comparison.append({
                "metric_name": metric_name,
                "model_a_value": val_a,
                "model_b_value": val_b,
                "delta": round(delta, 6),
                "improved": abs(val_b) < abs(val_a),  # Closer to 0 = more fair for difference metrics
            })

        return {
            "model_a": {
                "model_id": str(model_a_assessment.get("model_id", "")),
                "assessment_id": str(model_a_assessment.get("id", "")),
                "metrics": metrics_a,
            },
            "model_b": {
                "model_id": str(model_b_assessment.get("model_id", "")),
                "assessment_id": str(model_b_assessment.get("id", "")),
                "metrics": metrics_b,
            },
            "comparison": comparison,
        }

    def _build_group_rows(
        self,
        bias_metrics: list[dict[str, Any]],
        thresholds: dict[str, float],
    ) -> list[GroupMetricRow]:
        """Build per-group metric breakdown rows.

        Args:
            bias_metrics: BiasMetric records for an assessment.
            thresholds: Metric thresholds for pass/fail.

        Returns:
            List of GroupMetricRow objects.
        """
        group_metric_map: dict[str, dict[str, dict[str, Any]]] = {}

        for m in bias_metrics:
            group = m.get("group_label") or m.get("protected_attribute", "all")
            metric_name = m.get("metric_name", "unknown")
            value = float(m.get("value", 0.0))
            threshold = thresholds.get(metric_name, 0.1)
            is_ratio_metric = "ratio" in metric_name or metric_name == "disparate_impact"
            passed = value >= threshold if is_ratio_metric else abs(value) <= threshold

            if group not in group_metric_map:
                group_metric_map[group] = {}

            group_metric_map[group][metric_name] = {
                "value": round(value, 6),
                "threshold": threshold,
                "passed": passed,
            }

        rows: list[GroupMetricRow] = []
        for group_label, metrics in sorted(group_metric_map.items()):
            failed_metrics = sum(1 for m in metrics.values() if not m["passed"])
            n_metrics = len(metrics)
            overall_passed = failed_metrics == 0

            if failed_metrics == 0:
                risk_level = "low"
            elif failed_metrics / max(n_metrics, 1) < 0.5:
                risk_level = "medium"
            else:
                risk_level = "high"

            rows.append(GroupMetricRow(
                group_label=group_label,
                group_attributes={},  # Populated by repository layer
                n_samples=0,          # Populated by repository layer
                metrics=metrics,
                overall_passed=overall_passed,
                risk_level=risk_level,
            ))

        return rows

    def to_dict(self, dashboard: FairnessDashboard) -> dict[str, Any]:
        """Serialize a FairnessDashboard to a JSON-compatible dict.

        Args:
            dashboard: FairnessDashboard instance.

        Returns:
            JSON-serializable dict.
        """
        return {
            "model_id": dashboard.model_id,
            "assessment_id": dashboard.assessment_id,
            "overall_passed": dashboard.overall_passed,
            "pass_rate": dashboard.pass_rate,
            "assessment_timestamp": dashboard.assessment_timestamp,
            "protected_attributes": dashboard.protected_attributes,
            "metric_summaries": [
                {
                    "metric_name": ms.metric_name,
                    "value": ms.value,
                    "threshold": ms.threshold,
                    "passed": ms.passed,
                    "worst_group": ms.worst_group,
                    "best_group": ms.best_group,
                }
                for ms in dashboard.metric_summaries
            ],
            "group_rows": [
                {
                    "group_label": row.group_label,
                    "group_attributes": row.group_attributes,
                    "n_samples": row.n_samples,
                    "metrics": row.metrics,
                    "overall_passed": row.overall_passed,
                    "risk_level": row.risk_level,
                }
                for row in dashboard.group_rows
            ],
        }


__all__ = [
    "FairnessDashboardService",
    "FairnessDashboard",
    "MetricSummary",
    "GroupMetricRow",
    "TemporalDataPoint",
    "HeatmapCell",
]
