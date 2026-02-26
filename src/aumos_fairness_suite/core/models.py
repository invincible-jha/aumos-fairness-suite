"""SQLAlchemy ORM models for aumos-fairness-suite.

All tables use the `fai_` prefix. All models extend AumOSModel which provides
id (UUID PK), tenant_id (UUID FK), created_at, and updated_at columns.
Row-level security is enforced on all tenant-scoped tables.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Boolean, Float, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from aumos_common.database import AumOSModel


class FairnessAssessment(AumOSModel):
    """Records a fairness assessment run for a model and dataset.

    Stores the assessment configuration and aggregated metric results in JSONB.
    Individual metric rows are in BiasMetric (FK: fai_bias_metrics.assessment_id).

    Attributes:
        model_id: UUID string referencing the model in aumos-model-registry.
        dataset_id: UUID string referencing the dataset in aumos-data-layer.
        protected_attributes: JSONB list of ProtectedAttributeConfig dicts.
        label_column: Target/label column name used during assessment.
        prediction_column: Model prediction column name (nullable if label-only assessment).
        metrics: JSONB list of computed BiasMetricResult dicts (populated on completion).
        status: Current lifecycle status (pending/running/passed/failed/error).
    """

    __tablename__ = "fai_assessments"

    model_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    dataset_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    protected_attributes: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    label_column: Mapped[str] = mapped_column(String(255), nullable=False)
    prediction_column: Mapped[str | None] = mapped_column(String(255), nullable=True)
    metrics: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending", index=True)


class BiasMetric(AumOSModel):
    """Individual bias metric result within a fairness assessment.

    One row per metric per protected attribute per assessment. Enables querying
    and aggregating metric trends over time without parsing JSONB arrays.

    Attributes:
        assessment_id: UUID string FK to fai_assessments (cross-row, not FK constraint).
        metric_name: Standardised metric identifier (e.g. "disparate_impact").
        value: Computed numeric metric value.
        threshold: Pass/fail threshold applied.
        passed: Whether the metric satisfies its threshold.
        protected_attribute: Name of the protected attribute column assessed.
    """

    __tablename__ = "fai_bias_metrics"

    assessment_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    protected_attribute: Mapped[str] = mapped_column(String(255), nullable=False)


class MitigationJob(AumOSModel):
    """Records a debiasing job applied to a model or dataset.

    Captures before/after bias metrics to demonstrate the effectiveness of the
    applied mitigation strategy. Used in regulatory reports.

    Attributes:
        assessment_id: UUID string FK to the assessment that triggered this job.
        strategy: Mitigation stage: "pre", "in", or "post".
        algorithm: Specific algorithm applied (e.g. "reweighting", "threshold_optimization").
        status: Current job lifecycle status.
        before_metrics: JSONB dict of metric name -> value before mitigation.
        after_metrics: JSONB dict of metric name -> value after mitigation.
        error_message: Human-readable error if status is "failed".
    """

    __tablename__ = "fai_mitigation_jobs"

    assessment_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    strategy: Mapped[str] = mapped_column(String(50), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending", index=True)
    before_metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    after_metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class FairnessMonitor(AumOSModel):
    """Configuration for a continuous fairness monitoring subscription.

    The monitoring scheduler reads active monitors and triggers periodic
    assessments on a cron schedule. Alerts are published to Kafka when metrics
    exceed the configured threshold.

    Attributes:
        model_id: UUID string referencing the deployed model to monitor.
        protected_attributes: JSONB list of ProtectedAttributeConfig dicts.
        schedule_cron: Cron expression for the monitoring schedule.
        alert_threshold: Maximum allowed metric deviation before alerting.
        status: Monitor status: "active", "paused", or "disabled".
    """

    __tablename__ = "fai_monitors"

    model_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    protected_attributes: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    schedule_cron: Mapped[str] = mapped_column(String(100), nullable=False)
    alert_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.1)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="active", index=True)
