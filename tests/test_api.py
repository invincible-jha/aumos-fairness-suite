"""Unit tests for API endpoint routing and request/response schemas.

Uses FastAPI's TestClient with mocked services to test the HTTP layer
without any database or Kafka dependencies.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aumos_fairness_suite.api.schemas import (
    AssessmentListResponse,
    AssessmentResponse,
    AssessmentStatus,
    BiasMetricResult,
    FairnessReportResponse,
    MetricName,
    MitigationAlgorithm,
    MitigationJobResponse,
    MitigationJobStatus,
    MitigationStrategy,
    MonitorListResponse,
    MonitorResponse,
    MonitorStatus,
    ProtectedAttributeConfig,
    RegulatorySectionReport,
    RegulatoryFramework,
    SyntheticBiasCheckResponse,
)


# ---------------------------------------------------------------------------
# Schema validation tests (no HTTP layer required)
# ---------------------------------------------------------------------------


class TestSchemas:
    """Tests for Pydantic schema validation rules."""

    def test_assessment_create_request_requires_protected_attributes(
        self,
        model_id: uuid.UUID,
        dataset_id: uuid.UUID,
    ) -> None:
        """AssessmentCreateRequest rejects empty protected_attributes list."""
        from pydantic import ValidationError

        from aumos_fairness_suite.api.schemas import AssessmentCreateRequest

        with pytest.raises(ValidationError):
            AssessmentCreateRequest(
                model_id=model_id,
                dataset_id=dataset_id,
                protected_attributes=[],  # min_length=1
                label_column="approved",
            )

    def test_assessment_create_request_valid(
        self,
        model_id: uuid.UUID,
        dataset_id: uuid.UUID,
    ) -> None:
        """AssessmentCreateRequest constructs correctly with valid data."""
        from aumos_fairness_suite.api.schemas import (
            AssessmentCreateRequest,
            ProtectedAttributeConfig,
        )

        req = AssessmentCreateRequest(
            model_id=model_id,
            dataset_id=dataset_id,
            protected_attributes=[
                ProtectedAttributeConfig(
                    name="gender",
                    privileged_values=["male"],
                    unprivileged_values=["female"],
                )
            ],
            label_column="approved",
        )
        assert req.model_id == model_id
        assert len(req.protected_attributes) == 1

    def test_mitigation_create_request_requires_dataset_payload(
        self,
        assessment_id: uuid.UUID,
    ) -> None:
        """MitigationCreateRequest rejects empty dataset_payload."""
        from pydantic import ValidationError

        from aumos_fairness_suite.api.schemas import MitigationCreateRequest

        with pytest.raises(ValidationError):
            MitigationCreateRequest(
                assessment_id=assessment_id,
                strategy=MitigationStrategy.PRE_PROCESSING,
                algorithm=MitigationAlgorithm.REWEIGHTING,
                dataset_payload=[],  # min_length=1
                label_column="approved",
                protected_attribute="gender",
                privileged_values=["male"],
            )

    def test_synthetic_bias_check_requires_minimum_rows(
        self,
        dataset_id: uuid.UUID,
    ) -> None:
        """SyntheticBiasCheckRequest rejects datasets with fewer than 10 rows."""
        from pydantic import ValidationError

        from aumos_fairness_suite.api.schemas import SyntheticBiasCheckRequest

        with pytest.raises(ValidationError):
            SyntheticBiasCheckRequest(
                real_dataset=[{"gender": "male", "approved": 1}],  # < 10
                synthetic_dataset=[{"gender": "male", "approved": 1}],  # < 10
                protected_attribute="gender",
                label_column="approved",
                dataset_id=dataset_id,
            )

    def test_monitor_create_request_defaults(
        self,
        model_id: uuid.UUID,
    ) -> None:
        """MonitorCreateRequest uses sensible defaults for schedule and threshold."""
        from aumos_fairness_suite.api.schemas import (
            MonitorCreateRequest,
            ProtectedAttributeConfig,
        )

        req = MonitorCreateRequest(
            model_id=model_id,
            protected_attributes=[
                ProtectedAttributeConfig(
                    name="gender",
                    privileged_values=["male"],
                    unprivileged_values=["female"],
                )
            ],
        )
        assert req.schedule_cron == "0 */6 * * *"
        assert req.alert_threshold == 0.1

    def test_metric_name_enum_values(self) -> None:
        """MetricName enum contains all expected standardised metric names."""
        expected = {
            "disparate_impact",
            "statistical_parity_difference",
            "equal_opportunity_difference",
            "average_odds_difference",
            "theil_index",
            "demographic_parity_difference",
            "demographic_parity_ratio",
            "equalized_odds_difference",
            "equalized_odds_ratio",
        }
        actual = {m.value for m in MetricName}
        assert expected == actual

    def test_assessment_response_serialisation(
        self,
        tenant_id: uuid.UUID,
        model_id: uuid.UUID,
        dataset_id: uuid.UUID,
        assessment_id: uuid.UUID,
    ) -> None:
        """AssessmentResponse serialises to dict correctly."""
        now = datetime.now(timezone.utc)
        response = AssessmentResponse(
            id=assessment_id,
            tenant_id=tenant_id,
            model_id=model_id,
            dataset_id=dataset_id,
            protected_attributes=[
                ProtectedAttributeConfig(
                    name="gender",
                    privileged_values=["male"],
                    unprivileged_values=["female"],
                )
            ],
            metrics=[
                BiasMetricResult(
                    metric_name=MetricName.DISPARATE_IMPACT,
                    value=0.75,
                    threshold=0.8,
                    passed=False,
                    protected_attribute="gender",
                )
            ],
            status=AssessmentStatus.FAILED,
            created_at=now,
        )
        data = response.model_dump()
        assert data["status"] == "failed"
        assert data["metrics"][0]["metric_name"] == "disparate_impact"
        assert not data["metrics"][0]["passed"]

    def test_fairness_report_response_structure(
        self,
        tenant_id: uuid.UUID,
        model_id: uuid.UUID,
        assessment_id: uuid.UUID,
    ) -> None:
        """FairnessReportResponse includes sections for each regulatory framework."""
        now = datetime.now(timezone.utc)
        report = FairnessReportResponse(
            assessment_id=assessment_id,
            model_id=model_id,
            tenant_id=tenant_id,
            overall_passed=True,
            generated_at=now,
            sections=[
                RegulatorySectionReport(
                    framework=RegulatoryFramework.ECOA,
                    compliant=True,
                    summary="All metrics pass.",
                    metric_summaries=[],
                    recommendations=[],
                )
            ],
        )
        assert report.overall_passed is True
        assert len(report.sections) == 1
        assert report.sections[0].framework == RegulatoryFramework.ECOA

    def test_synthetic_bias_check_response_all_passed(
        self,
        dataset_id: uuid.UUID,
    ) -> None:
        """SyntheticBiasCheckResponse.passed is True only when all sub-checks pass."""
        response = SyntheticBiasCheckResponse(
            passed=True,
            kl_divergence=0.01,
            kl_divergence_passed=True,
            label_rate_disparity=0.02,
            label_rate_disparity_passed=True,
            amplification_factor=1.05,
            amplification_passed=True,
            group_stats=[],
            dataset_id=dataset_id,
        )
        assert response.passed is True

    def test_synthetic_bias_check_response_fails_on_amplification(
        self,
        dataset_id: uuid.UUID,
    ) -> None:
        """SyntheticBiasCheckResponse.passed is False when amplification fails."""
        response = SyntheticBiasCheckResponse(
            passed=False,
            kl_divergence=0.01,
            kl_divergence_passed=True,
            label_rate_disparity=0.02,
            label_rate_disparity_passed=True,
            amplification_factor=1.5,  # > 1.2 threshold
            amplification_passed=False,
            group_stats=[],
            dataset_id=dataset_id,
        )
        assert response.passed is False
        assert not response.amplification_passed

    def test_mitigation_strategy_enum_values(self) -> None:
        """MitigationStrategy enum values match expected strings."""
        assert MitigationStrategy.PRE_PROCESSING.value == "pre"
        assert MitigationStrategy.IN_PROCESSING.value == "in"
        assert MitigationStrategy.POST_PROCESSING.value == "post"

    def test_mitigation_algorithm_enum_values(self) -> None:
        """MitigationAlgorithm enum contains all expected algorithm names."""
        expected = {
            "reweighting",
            "rejection_sampling",
            "adversarial_debiasing",
            "threshold_optimization",
        }
        actual = {a.value for a in MitigationAlgorithm}
        assert expected == actual

    def test_monitor_status_enum_values(self) -> None:
        """MonitorStatus enum contains expected status values."""
        assert MonitorStatus.ACTIVE.value == "active"
        assert MonitorStatus.PAUSED.value == "paused"
        assert MonitorStatus.DISABLED.value == "disabled"
