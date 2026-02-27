"""Abstract interfaces (Protocol classes) for the Fairness Suite.

Services depend on these protocols, not on concrete adapter implementations.
This enables unit testing with mock adapters and future adapter swapping.

Protocols defined:
- BiasDetector
- MitigationAdapter
- SyntheticBiasDetector
- FairnessEventPublisher
- IIntersectionalAnalyzer
- IRemediationAdvisor
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BiasDetector(Protocol):
    """Protocol for bias metric computation adapters.

    Concrete implementations include AIF360Detector and FairlearnDetector.
    """

    @property
    def supported_metrics(self) -> list[str]:
        """Return the list of metric names this detector can compute.

        Returns:
            List of standardised metric name strings.
        """
        ...

    def compute_metrics(
        self,
        features: list[dict[str, Any]],
        labels: list[int],
        predictions: list[int] | None,
        protected_attribute: str,
        privileged_values: list[Any],
        unprivileged_values: list[Any],
    ) -> dict[str, float]:
        """Compute bias metrics for a dataset.

        All computation is synchronous (CPU-bound). Call via asyncio.to_thread()
        in async service methods.

        Args:
            features: List of feature dicts (one per sample).
            labels: Ground-truth binary labels (0 or 1).
            predictions: Model predictions (0 or 1). None for label-only metrics.
            protected_attribute: Column name to assess fairness on.
            privileged_values: Values belonging to the privileged group.
            unprivileged_values: Values belonging to the unprivileged group.

        Returns:
            Dict mapping metric_name -> computed float value.
        """
        ...


@runtime_checkable
class MitigationAdapter(Protocol):
    """Protocol for debiasing algorithm adapters.

    Concrete implementations exist per strategy stage in adapters/mitigation/.
    """

    @property
    def supported_algorithms(self) -> list[str]:
        """Return the list of algorithm names this adapter can execute.

        Returns:
            List of algorithm name strings.
        """
        ...

    def apply(
        self,
        features: list[dict[str, Any]],
        labels: list[int],
        protected_attribute: str,
        privileged_values: list[Any],
        algorithm: str,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
        """Apply debiasing to a dataset.

        Args:
            features: Feature dicts (one per sample).
            labels: Binary labels (0 or 1).
            protected_attribute: Column name to debias on.
            privileged_values: Values belonging to the privileged group.
            algorithm: Which algorithm variant to apply.
            **kwargs: Algorithm-specific parameters (e.g. fairness_constraint).

        Returns:
            Tuple of (transformed_features, transformed_labels, metadata_dict).
            Pre-processing adapters return modified data.
            Post-processing adapters return the original data with a metadata dict
            containing decision thresholds per group.
        """
        ...


@runtime_checkable
class SyntheticBiasDetector(Protocol):
    """Protocol for synthetic data bias amplification detectors."""

    def check_amplification(
        self,
        real_features: list[dict[str, Any]],
        real_labels: list[int],
        synthetic_features: list[dict[str, Any]],
        synthetic_labels: list[int],
        protected_attribute: str,
    ) -> dict[str, Any]:
        """Check whether synthetic data amplifies bias from real data.

        Args:
            real_features: Feature dicts for the real training dataset.
            real_labels: Labels for the real dataset.
            synthetic_features: Feature dicts for the synthetic dataset.
            synthetic_labels: Labels for the synthetic dataset.
            protected_attribute: Column name of the protected attribute.

        Returns:
            Dict with keys: kl_divergence, label_rate_disparity,
            amplification_factor, group_stats.
        """
        ...


@runtime_checkable
class FairnessEventPublisher(Protocol):
    """Protocol for publishing fairness lifecycle events to Kafka."""

    async def publish_assessment_complete(
        self,
        tenant_id: str,
        assessment_id: str,
        model_id: str,
        passed: bool,
    ) -> None:
        """Publish fairness.assessment_complete (or fairness.assessment_failed) event.

        Args:
            tenant_id: Owning tenant UUID string.
            assessment_id: Assessment UUID string.
            model_id: Model UUID string.
            passed: True if all metrics passed, False if any failed.
        """
        ...

    async def publish_monitor_alert(
        self,
        tenant_id: str,
        monitor_id: str,
        model_id: str,
        metric_name: str,
        metric_value: float,
        threshold: float,
    ) -> None:
        """Publish fairness.monitor_alert event when a metric breaches threshold.

        Args:
            tenant_id: Owning tenant UUID string.
            monitor_id: Monitor UUID string.
            model_id: Model UUID string.
            metric_name: Name of the metric that breached the threshold.
            metric_value: Observed metric value.
            threshold: Configured alert threshold.
        """
        ...

    async def publish_synthetic_bias_detected(
        self,
        tenant_id: str,
        dataset_id: str | None,
        amplification_factor: float,
    ) -> None:
        """Publish fairness.synthetic_bias_detected event.

        Args:
            tenant_id: Owning tenant UUID string.
            dataset_id: Synthetic dataset UUID string (may be None).
            amplification_factor: Computed amplification factor.
        """
        ...


@runtime_checkable
class IIntersectionalAnalyzer(Protocol):
    """Protocol for multi-dimensional intersectional fairness analysis.

    Concrete implementation: IntersectionalAnalyzer in adapters/intersectional_analyzer.py.
    Satisfies BiasDetector protocol (is a specialised BiasDetector).
    """

    @property
    def supported_metrics(self) -> list[str]:
        """Return the list of intersectional metric names this analyzer computes.

        Returns:
            List of metric name strings (e.g., 'intersectional_tpr_gap').
        """
        ...

    def compute_metrics(
        self,
        features: list[dict[str, Any]],
        labels: list[int],
        predictions: list[int] | None,
        protected_attribute: str,
        privileged_values: list[Any],
        unprivileged_values: list[Any],
    ) -> dict[str, float]:
        """Compute intersectional bias metrics for a dataset (BiasDetector compat).

        Args:
            features: List of feature dicts (one per sample).
            labels: Ground-truth binary labels.
            predictions: Model predictions. None for label-only metrics.
            protected_attribute: Primary column name to assess fairness on.
            privileged_values: Values belonging to the privileged group.
            unprivileged_values: Values belonging to the unprivileged group.

        Returns:
            Dict mapping metric_name -> computed float value.
        """
        ...

    def analyze_intersectional(
        self,
        features: list[dict[str, Any]],
        labels: list[int],
        predictions: list[int],
        protected_attributes: list[str],
        max_combination_depth: int,
    ) -> dict[str, Any]:
        """Run full intersectional analysis across attribute combinations.

        Args:
            features: Feature dicts (one per sample).
            labels: Ground-truth binary labels.
            predictions: Model predictions.
            protected_attributes: Column names to analyze intersectionally.
            max_combination_depth: Maximum number of attributes to combine.

        Returns:
            Dict with intersectional_amplification, disparate_groups,
            severity_distribution, regulatory_flags, and visualization_data.
        """
        ...


@runtime_checkable
class IRemediationAdvisor(Protocol):
    """Protocol for bias remediation strategy recommendation.

    Concrete implementation: RemediationAdvisor in adapters/remediation_advisor.py.
    """

    def advise(
        self,
        bias_metrics: dict[str, float],
        feature_contributions: dict[str, float] | None,
        intersectional_result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Generate a prioritised remediation roadmap from bias analysis results.

        Args:
            bias_metrics: Dict of metric_name -> computed value.
            feature_contributions: Optional dict of feature_name -> bias attribution.
            intersectional_result: Optional intersectional analysis result dict.

        Returns:
            Dict with recommended_strategies (list), roadmap (phased dict),
            projected_improvements, and compliance_guidance.
        """
        ...

    def generate_feature_remediation(
        self,
        feature_name: str,
        bias_attribution_score: float,
        data_type: str,
    ) -> dict[str, Any]:
        """Generate per-feature bias remediation recommendations.

        Args:
            feature_name: Name of the biased feature.
            bias_attribution_score: Attribution score (higher = more biased).
            data_type: 'categorical' or 'numerical'.

        Returns:
            Dict with feature, severity, recommendations, and implementation_steps.
        """
        ...
