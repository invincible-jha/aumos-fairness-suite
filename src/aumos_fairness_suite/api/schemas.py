"""Pydantic request and response schemas for the Fairness Suite API."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AssessmentStatus(str, Enum):
    """Lifecycle status of a fairness assessment."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


class MitigationStrategy(str, Enum):
    """High-level mitigation stage — when in the pipeline debiasing is applied."""

    PRE_PROCESSING = "pre"
    IN_PROCESSING = "in"
    POST_PROCESSING = "post"


class MitigationAlgorithm(str, Enum):
    """Concrete debiasing algorithms available per strategy stage."""

    # Pre-processing
    REWEIGHTING = "reweighting"
    REJECTION_SAMPLING = "rejection_sampling"
    # In-processing
    ADVERSARIAL_DEBIASING = "adversarial_debiasing"
    # Post-processing
    THRESHOLD_OPTIMIZATION = "threshold_optimization"


class MitigationJobStatus(str, Enum):
    """Lifecycle status of a mitigation job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class MonitorStatus(str, Enum):
    """Whether a continuous fairness monitor is active."""

    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"


class MetricName(str, Enum):
    """Standardised metric identifiers across AIF360 and Fairlearn."""

    # AIF360
    DISPARATE_IMPACT = "disparate_impact"
    STATISTICAL_PARITY_DIFFERENCE = "statistical_parity_difference"
    EQUAL_OPPORTUNITY_DIFFERENCE = "equal_opportunity_difference"
    AVERAGE_ODDS_DIFFERENCE = "average_odds_difference"
    THEIL_INDEX = "theil_index"
    # Fairlearn
    DEMOGRAPHIC_PARITY_DIFFERENCE = "demographic_parity_difference"
    DEMOGRAPHIC_PARITY_RATIO = "demographic_parity_ratio"
    EQUALIZED_ODDS_DIFFERENCE = "equalized_odds_difference"
    EQUALIZED_ODDS_RATIO = "equalized_odds_ratio"


class RegulatoryFramework(str, Enum):
    """Supported regulatory frameworks for fairness reports."""

    ECOA = "ECOA"
    EU_AI_ACT = "EU_AI_ACT"


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


class ProtectedAttributeConfig(BaseModel):
    """Configuration for a single protected attribute in an assessment.

    Args:
        name: Column name of the protected attribute in the dataset.
        privileged_values: List of values that belong to the privileged group.
        unprivileged_values: List of values that belong to the unprivileged group.
    """

    name: str
    privileged_values: list[Any]
    unprivileged_values: list[Any]


class BiasMetricResult(BaseModel):
    """Result for a single bias metric within an assessment.

    Args:
        metric_name: Standardised metric identifier.
        value: Computed numeric value.
        threshold: Pass/fail threshold used.
        passed: Whether the metric value satisfies the threshold.
        protected_attribute: Which protected attribute this metric was computed for.
    """

    metric_name: MetricName
    value: float
    threshold: float
    passed: bool
    protected_attribute: str


# ---------------------------------------------------------------------------
# Assessment schemas
# ---------------------------------------------------------------------------


class AssessmentCreateRequest(BaseModel):
    """Request to run a fairness assessment on a model and dataset.

    Args:
        model_id: UUID of the deployed model to assess.
        dataset_id: UUID of the dataset to use for assessment.
        protected_attributes: One entry per protected attribute to assess.
        label_column: Name of the target/label column in the dataset.
        prediction_column: Name of the model prediction column. If omitted,
            only dataset-level metrics (disparate impact on labels) are computed.
        metric_names: Which metrics to compute. Defaults to all standard metrics.
    """

    model_id: uuid.UUID
    dataset_id: uuid.UUID
    protected_attributes: list[ProtectedAttributeConfig] = Field(min_length=1)
    label_column: str
    prediction_column: str | None = None
    metric_names: list[MetricName] | None = None


class AssessmentResponse(BaseModel):
    """Response for a created or retrieved fairness assessment.

    Args:
        id: Assessment UUID.
        tenant_id: Owning tenant UUID.
        model_id: Assessed model UUID.
        dataset_id: Dataset used for assessment.
        protected_attributes: Protected attribute configurations (serialised).
        metrics: Computed metric results (populated after assessment completes).
        status: Current lifecycle status.
        created_at: Timestamp of assessment creation.
        updated_at: Timestamp of last status change.
    """

    id: uuid.UUID
    tenant_id: uuid.UUID
    model_id: uuid.UUID
    dataset_id: uuid.UUID
    protected_attributes: list[ProtectedAttributeConfig]
    metrics: list[BiasMetricResult] = Field(default_factory=list)
    status: AssessmentStatus
    created_at: datetime
    updated_at: datetime | None = None


class AssessmentListResponse(BaseModel):
    """Paginated list of fairness assessments."""

    items: list[AssessmentResponse]
    total: int
    page: int
    page_size: int


# ---------------------------------------------------------------------------
# Mitigation schemas
# ---------------------------------------------------------------------------


class MitigationCreateRequest(BaseModel):
    """Request to apply a debiasing strategy to a model or dataset.

    Args:
        assessment_id: UUID of the failed assessment that triggered this mitigation.
        strategy: Pre-, in-, or post-processing stage.
        algorithm: Specific debiasing algorithm to apply.
        dataset_payload: Raw dataset rows as a list of dicts (inline delivery —
            this service does not fetch datasets from external storage).
        label_column: Target column name in dataset_payload.
        protected_attribute: Name of the protected attribute to debias.
        privileged_values: Privileged group values for the protected attribute.
        fairness_constraint: Fairness criterion for post-processing threshold
            optimisation (e.g. "equalized_odds", "demographic_parity").
    """

    assessment_id: uuid.UUID
    strategy: MitigationStrategy
    algorithm: MitigationAlgorithm
    dataset_payload: list[dict[str, Any]] = Field(min_length=1)
    label_column: str
    protected_attribute: str
    privileged_values: list[Any]
    fairness_constraint: str = "equalized_odds"


class MitigationJobResponse(BaseModel):
    """Response for a mitigation job.

    Args:
        id: Job UUID.
        tenant_id: Owning tenant UUID.
        assessment_id: Assessment that triggered the job.
        strategy: Applied strategy stage.
        algorithm: Applied debiasing algorithm.
        status: Current job status.
        before_metrics: Bias metrics measured before mitigation.
        after_metrics: Bias metrics measured after mitigation (populated on completion).
        created_at: Job creation timestamp.
        completed_at: Job completion timestamp (None if still running).
    """

    id: uuid.UUID
    tenant_id: uuid.UUID
    assessment_id: uuid.UUID
    strategy: MitigationStrategy
    algorithm: MitigationAlgorithm
    status: MitigationJobStatus
    before_metrics: dict[str, float] = Field(default_factory=dict)
    after_metrics: dict[str, float] = Field(default_factory=dict)
    created_at: datetime
    completed_at: datetime | None = None


# ---------------------------------------------------------------------------
# Monitor schemas
# ---------------------------------------------------------------------------


class MonitorCreateRequest(BaseModel):
    """Request to create a continuous fairness monitor for a deployed model.

    Args:
        model_id: UUID of the model to monitor.
        protected_attributes: Protected attribute configurations to monitor.
        schedule_cron: Cron expression for the monitoring schedule (e.g. "0 */6 * * *").
        alert_threshold: Maximum allowed value for any difference metric before alerting.
            Uses the absolute value for bidirectional metrics.
    """

    model_id: uuid.UUID
    protected_attributes: list[ProtectedAttributeConfig] = Field(min_length=1)
    schedule_cron: str = "0 */6 * * *"
    alert_threshold: float = 0.1


class MonitorResponse(BaseModel):
    """Response for a fairness monitor.

    Args:
        id: Monitor UUID.
        tenant_id: Owning tenant UUID.
        model_id: Monitored model UUID.
        protected_attributes: Protected attribute configurations.
        schedule_cron: Monitoring schedule cron expression.
        alert_threshold: Alert trigger threshold.
        status: Monitor status.
        created_at: Creation timestamp.
    """

    id: uuid.UUID
    tenant_id: uuid.UUID
    model_id: uuid.UUID
    protected_attributes: list[ProtectedAttributeConfig]
    schedule_cron: str
    alert_threshold: float
    status: MonitorStatus
    created_at: datetime


class MonitorListResponse(BaseModel):
    """Paginated list of fairness monitors."""

    items: list[MonitorResponse]
    total: int
    page: int
    page_size: int


# ---------------------------------------------------------------------------
# Synthetic bias check schemas
# ---------------------------------------------------------------------------


class SyntheticBiasCheckRequest(BaseModel):
    """Request to check synthetic data for bias amplification.

    Args:
        real_dataset: Real training dataset rows (list of dicts).
        synthetic_dataset: Synthetic dataset rows (list of dicts) from a generative model.
        protected_attribute: Column name of the protected attribute to analyse.
        label_column: Target/label column name.
        dataset_id: Optional UUID of the synthetic dataset in aumos-data-layer, for audit.
    """

    real_dataset: list[dict[str, Any]] = Field(min_length=10)
    synthetic_dataset: list[dict[str, Any]] = Field(min_length=10)
    protected_attribute: str
    label_column: str
    dataset_id: uuid.UUID | None = None


class GroupStats(BaseModel):
    """Aggregate statistics for one protected attribute group.

    Args:
        group_value: The protected attribute group value (e.g. "female", 0, "hispanic").
        real_count: Number of instances in this group in the real dataset.
        synthetic_count: Number of instances in this group in the synthetic dataset.
        real_positive_rate: Fraction of positive labels in this group in the real data.
        synthetic_positive_rate: Fraction of positive labels in this group in synthetic data.
    """

    group_value: Any
    real_count: int
    synthetic_count: int
    real_positive_rate: float
    synthetic_positive_rate: float


class SyntheticBiasCheckResponse(BaseModel):
    """Response for a synthetic bias amplification check.

    Args:
        passed: Whether all checks passed (no significant amplification detected).
        kl_divergence: KL divergence of group distribution: real vs synthetic.
        kl_divergence_passed: Whether KL divergence is below the threshold.
        label_rate_disparity: Max absolute label rate difference across groups.
        label_rate_disparity_passed: Whether label rate disparity is below threshold.
        amplification_factor: Ratio of disparate impact in synthetic vs real data.
            A value > 1.0 means synthetic data has more bias than real data.
        amplification_passed: Whether amplification factor is below threshold.
        group_stats: Per-group aggregate statistics.
        dataset_id: Echo of the input dataset_id for audit linkage.
    """

    passed: bool
    kl_divergence: float
    kl_divergence_passed: bool
    label_rate_disparity: float
    label_rate_disparity_passed: bool
    amplification_factor: float
    amplification_passed: bool
    group_stats: list[GroupStats]
    dataset_id: uuid.UUID | None = None


# ---------------------------------------------------------------------------
# Regulatory report schemas
# ---------------------------------------------------------------------------


class MetricSummary(BaseModel):
    """Summary of a single metric for a regulatory report section."""

    metric_name: str
    value: float
    threshold: float
    passed: bool
    interpretation: str


class RegulatorySectionReport(BaseModel):
    """Report section covering one regulatory framework's requirements."""

    framework: RegulatoryFramework
    compliant: bool
    summary: str
    metric_summaries: list[MetricSummary]
    recommendations: list[str]


class FairnessReportResponse(BaseModel):
    """Full regulatory fairness report for an assessment.

    Args:
        assessment_id: UUID of the underlying assessment.
        model_id: Assessed model UUID.
        tenant_id: Owning tenant UUID.
        overall_passed: Whether the model passes all applicable regulatory requirements.
        generated_at: Report generation timestamp.
        sections: One section per applicable regulatory framework.
        mitigation_history: Summary of any mitigation jobs applied since the assessment.
    """

    assessment_id: uuid.UUID
    model_id: uuid.UUID
    tenant_id: uuid.UUID
    overall_passed: bool
    generated_at: datetime
    sections: list[RegulatorySectionReport]
    mitigation_history: list[dict[str, Any]] = Field(default_factory=list)
