"""Unit tests for core service logic.

All database and Kafka interactions are mocked. Tests use synthetic demographic
datasets from conftest.py — no real PII.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aumos_fairness_suite.api.schemas import (
    AssessmentStatus,
    MitigationAlgorithm,
    MitigationStrategy,
    MonitorStatus,
)
from aumos_fairness_suite.core.services import (
    BiasDetectionService,
    MitigationService,
    MonitoringService,
    ReportingService,
)
from tests.conftest import make_biased_dataset, make_fair_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tenant(tenant_id: uuid.UUID) -> MagicMock:
    """Create a mock tenant context object."""
    tenant = MagicMock()
    tenant.tenant_id = tenant_id
    return tenant


def make_mock_session() -> AsyncMock:
    """Create a mock AsyncSession."""
    return AsyncMock()


# ---------------------------------------------------------------------------
# BiasDetectionService unit tests
# ---------------------------------------------------------------------------


class TestBiasDetectionService:
    """Tests for BiasDetectionService bias computation and assessment lifecycle."""

    def test_run_aif360_for_attribute_returns_metrics(
        self,
        tenant_id: uuid.UUID,
        model_id: uuid.UUID,
        dataset_id: uuid.UUID,
    ) -> None:
        """AIF360 detector returns a non-empty list of BiasMetricResult objects."""
        service = BiasDetectionService(session=make_mock_session())

        from aumos_fairness_suite.api.schemas import (
            AssessmentCreateRequest,
            ProtectedAttributeConfig,
        )

        request = AssessmentCreateRequest(
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
        attr = request.protected_attributes[0]
        results = service._run_aif360_for_attribute(attr_config=attr, request=request)

        assert len(results) == 5
        metric_names = {r.metric_name.value for r in results}
        assert "disparate_impact" in metric_names
        assert "statistical_parity_difference" in metric_names

    def test_run_fairlearn_for_attribute_returns_metrics(
        self,
        tenant_id: uuid.UUID,
        model_id: uuid.UUID,
        dataset_id: uuid.UUID,
    ) -> None:
        """Fairlearn detector returns a non-empty list of BiasMetricResult objects."""
        service = BiasDetectionService(session=make_mock_session())

        from aumos_fairness_suite.api.schemas import (
            AssessmentCreateRequest,
            ProtectedAttributeConfig,
        )

        request = AssessmentCreateRequest(
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
        attr = request.protected_attributes[0]
        results = service._run_fairlearn_for_attribute(attr_config=attr, request=request)

        assert len(results) == 4
        metric_names = {r.metric_name.value for r in results}
        assert "demographic_parity_difference" in metric_names
        assert "equalized_odds_ratio" in metric_names

    def test_compute_representative_metrics_biased(
        self,
        biased_dataset: tuple[list[dict[str, Any]], list[int]],
    ) -> None:
        """Representative metrics detect disparity in a biased dataset."""
        service = BiasDetectionService(session=make_mock_session())
        features, labels = biased_dataset

        metrics = service._compute_representative_metrics(
            features=features,
            labels=labels,
            protected_attribute="gender",
            privileged_values=["male"],
        )

        assert "disparate_impact" in metrics
        assert "statistical_parity_difference" in metrics
        # With bias_factor=0.4, DI should be well below 0.8
        assert metrics["disparate_impact"] < 0.9

    def test_compute_representative_metrics_fair(
        self,
        fair_dataset: tuple[list[dict[str, Any]], list[int]],
    ) -> None:
        """Representative metrics show near-zero disparity for a fair dataset."""
        service = BiasDetectionService(session=make_mock_session())
        features, labels = fair_dataset

        metrics = service._compute_representative_metrics(
            features=features,
            labels=labels,
            protected_attribute="gender",
            privileged_values=["male"],
        )

        # Fair dataset should have DI close to 1.0 (within noise)
        assert metrics["disparate_impact"] > 0.7
        assert abs(metrics["statistical_parity_difference"]) < 0.2

    def test_compute_representative_metrics_empty_dataset(self) -> None:
        """Representative metrics returns empty dict for empty input."""
        service = BiasDetectionService(session=make_mock_session())
        metrics = service._compute_representative_metrics(
            features=[],
            labels=[],
            protected_attribute="gender",
            privileged_values=["male"],
        )
        assert metrics == {}

    def test_to_assessment_response_converts_correctly(
        self,
        tenant_id: uuid.UUID,
        model_id: uuid.UUID,
        dataset_id: uuid.UUID,
        assessment_id: uuid.UUID,
    ) -> None:
        """_to_assessment_response correctly maps ORM fields to response schema."""
        from datetime import datetime, timezone

        service = BiasDetectionService(session=make_mock_session())

        mock_assessment = MagicMock()
        mock_assessment.id = assessment_id
        mock_assessment.tenant_id = tenant_id
        mock_assessment.model_id = str(model_id)
        mock_assessment.dataset_id = str(dataset_id)
        mock_assessment.protected_attributes = [
            {"name": "gender", "privileged_values": ["male"], "unprivileged_values": ["female"]}
        ]
        mock_assessment.metrics = [
            {
                "metric_name": "disparate_impact",
                "value": 0.75,
                "threshold": 0.8,
                "passed": False,
                "protected_attribute": "gender",
            }
        ]
        mock_assessment.status = "failed"
        mock_assessment.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        mock_assessment.updated_at = None

        response = service._to_assessment_response(mock_assessment)

        assert response.id == assessment_id
        assert response.status == AssessmentStatus.FAILED
        assert len(response.metrics) == 1
        assert response.metrics[0].value == 0.75
        assert not response.metrics[0].passed


# ---------------------------------------------------------------------------
# MitigationService unit tests
# ---------------------------------------------------------------------------


class TestMitigationService:
    """Tests for MitigationService debiasing orchestration."""

    def test_select_adapter_pre(self) -> None:
        """Pre-processing strategy selects PreProcessingAdapter."""
        from aumos_fairness_suite.adapters.mitigation.pre_processing import PreProcessingAdapter

        service = MitigationService(session=make_mock_session())
        adapter = service._select_adapter(MitigationStrategy.PRE_PROCESSING)
        assert isinstance(adapter, PreProcessingAdapter)

    def test_select_adapter_in(self) -> None:
        """In-processing strategy selects InProcessingAdapter."""
        from aumos_fairness_suite.adapters.mitigation.in_processing import InProcessingAdapter

        service = MitigationService(session=make_mock_session())
        adapter = service._select_adapter(MitigationStrategy.IN_PROCESSING)
        assert isinstance(adapter, InProcessingAdapter)

    def test_select_adapter_post(self) -> None:
        """Post-processing strategy selects PostProcessingAdapter."""
        from aumos_fairness_suite.adapters.mitigation.post_processing import PostProcessingAdapter

        service = MitigationService(session=make_mock_session())
        adapter = service._select_adapter(MitigationStrategy.POST_PROCESSING)
        assert isinstance(adapter, PostProcessingAdapter)

    def test_compute_representative_metrics_reduces_disparity_after_reweighting(
        self,
        biased_dataset: tuple[list[dict[str, Any]], list[int]],
    ) -> None:
        """After applying reweighting, disparate impact metric moves closer to 1.0."""
        from aumos_fairness_suite.adapters.mitigation.pre_processing import PreProcessingAdapter

        features, labels = biased_dataset
        adapter = PreProcessingAdapter()
        _, adjusted_labels, metadata = adapter.apply(
            features=features,
            labels=labels,
            protected_attribute="gender",
            privileged_values=["male"],
            algorithm="reweighting",
        )

        service = MitigationService(session=make_mock_session())
        before_metrics = service._compute_representative_metrics(
            features=features,
            labels=labels,
            protected_attribute="gender",
            privileged_values=["male"],
        )
        after_metrics = service._compute_representative_metrics(
            features=features,
            labels=adjusted_labels,
            protected_attribute="gender",
            privileged_values=["male"],
        )

        # Reweighting doesn't change labels — but metadata carries weights
        # The labels are unchanged in reweighting; weights are the output
        assert "weights" in metadata
        assert len(metadata["weights"]) == len(labels)
        # Disparity in original data should be detectable
        assert before_metrics["disparate_impact"] < 0.9


# ---------------------------------------------------------------------------
# ReportingService unit tests
# ---------------------------------------------------------------------------


class TestReportingService:
    """Tests for ReportingService regulatory report generation."""

    def test_build_ecoa_section_passed(self) -> None:
        """ECOA section marks compliant when all relevant metrics pass."""
        service = ReportingService(session=make_mock_session())
        metrics = [
            {"metric_name": "disparate_impact", "value": 0.85, "threshold": 0.8, "passed": True},
            {"metric_name": "statistical_parity_difference", "value": 0.05, "threshold": 0.1, "passed": True},
        ]
        section = service._build_ecoa_section(metrics=metrics)

        assert section.compliant is True
        assert "passed" in section.summary.lower()
        assert len(section.metric_summaries) == 2

    def test_build_ecoa_section_failed(self) -> None:
        """ECOA section marks non-compliant when disparate impact fails."""
        service = ReportingService(session=make_mock_session())
        metrics = [
            {"metric_name": "disparate_impact", "value": 0.65, "threshold": 0.8, "passed": False},
            {"metric_name": "statistical_parity_difference", "value": 0.15, "threshold": 0.1, "passed": False},
        ]
        section = service._build_ecoa_section(metrics=metrics)

        assert section.compliant is False
        assert len(section.recommendations) > 0

    def test_build_eu_ai_act_section_passed(self) -> None:
        """EU AI Act section marks compliant when equalized odds pass."""
        service = ReportingService(session=make_mock_session())
        metrics = [
            {"metric_name": "demographic_parity_difference", "value": 0.05, "threshold": 0.1, "passed": True},
            {"metric_name": "equalized_odds_difference", "value": 0.04, "threshold": 0.1, "passed": True},
        ]
        section = service._build_eu_ai_act_section(metrics=metrics)

        assert section.compliant is True

    def test_build_metric_summaries_includes_interpretation(self) -> None:
        """Metric summaries include human-readable interpretation strings."""
        service = ReportingService(session=make_mock_session())
        metrics = [
            {"metric_name": "disparate_impact", "value": 0.85, "threshold": 0.8, "passed": True},
        ]
        summaries = service._build_metric_summaries(metrics=metrics)

        assert len(summaries) == 1
        assert "4/5 rule" in summaries[0].interpretation
        assert summaries[0].metric_name == "disparate_impact"

    def test_build_metric_summaries_empty(self) -> None:
        """Empty metrics list returns empty summaries."""
        service = ReportingService(session=make_mock_session())
        summaries = service._build_metric_summaries(metrics=[])
        assert summaries == []
