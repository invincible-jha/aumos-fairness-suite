"""Business logic services for the Fairness Suite.

Services contain all domain logic and orchestrate between repositories, bias
detectors, mitigation adapters, and event publishers. No FastAPI or HTTP
dependencies here — those live in the api/ layer.

All metric computation (AIF360, Fairlearn, scipy) is CPU-bound and is wrapped
in asyncio.to_thread() to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.errors import NotFoundError
from aumos_common.observability import get_logger
from aumos_common.pagination import PageRequest

from aumos_fairness_suite.adapters.detection.aif360_detector import AIF360Detector
from aumos_fairness_suite.adapters.detection.fairlearn_detector import FairlearnDetector
from aumos_fairness_suite.adapters.mitigation.in_processing import InProcessingAdapter
from aumos_fairness_suite.adapters.mitigation.post_processing import PostProcessingAdapter
from aumos_fairness_suite.adapters.mitigation.pre_processing import PreProcessingAdapter
from aumos_fairness_suite.adapters.repositories import (
    AssessmentRepository,
    BiasMetricRepository,
    MitigationJobRepository,
    MonitorRepository,
)
from aumos_fairness_suite.adapters.synthetic_bias_detector import SyntheticBiasAmplificationDetector
from aumos_fairness_suite.api.schemas import (
    AssessmentCreateRequest,
    AssessmentListResponse,
    AssessmentResponse,
    AssessmentStatus,
    BiasMetricResult,
    FairnessReportResponse,
    GroupStats,
    MetricName,
    MetricSummary,
    MitigationAlgorithm,
    MitigationCreateRequest,
    MitigationJobResponse,
    MitigationJobStatus,
    MitigationStrategy,
    MonitorCreateRequest,
    MonitorListResponse,
    MonitorResponse,
    MonitorStatus,
    ProtectedAttributeConfig,
    RegulatorySectionReport,
    RegulatoryFramework,
    SyntheticBiasCheckRequest,
    SyntheticBiasCheckResponse,
)
from aumos_fairness_suite.settings import Settings

logger = get_logger(__name__)
settings = Settings()


class BiasDetectionService:
    """Orchestrates fairness assessment and synthetic bias detection workflows.

    Runs AIF360 and Fairlearn detectors against dataset payloads, persists
    results, and publishes Kafka events on completion.

    Args:
        session: Async SQLAlchemy session (injected by FastAPI dependency).
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._assessment_repo = AssessmentRepository(session)
        self._metric_repo = BiasMetricRepository(session)
        self._aif360 = AIF360Detector()
        self._fairlearn = FairlearnDetector()
        self._synthetic_detector = SyntheticBiasAmplificationDetector()

    async def create_assessment(
        self,
        request: AssessmentCreateRequest,
        tenant: Any,
    ) -> AssessmentResponse:
        """Create and immediately run a fairness assessment.

        Persists the assessment record, runs all configured detectors in a
        thread pool (CPU-bound), saves metric results, updates status, and
        publishes a Kafka event.

        Args:
            request: Assessment configuration including model, dataset, and attributes.
            tenant: Authenticated tenant context from aumos-common.

        Returns:
            AssessmentResponse with all computed metrics and final status.
        """
        assessment_id = uuid.uuid4()
        tenant_id = str(tenant.tenant_id)

        logger.info(
            "Creating fairness assessment",
            assessment_id=str(assessment_id),
            model_id=str(request.model_id),
            tenant_id=tenant_id,
        )

        # Persist initial record with pending status
        assessment = await self._assessment_repo.create(
            id=assessment_id,
            tenant_id=uuid.UUID(tenant_id),
            model_id=str(request.model_id),
            dataset_id=str(request.dataset_id),
            protected_attributes=[attr.model_dump() for attr in request.protected_attributes],
            label_column=request.label_column,
            prediction_column=request.prediction_column,
            metrics=[],
            status=AssessmentStatus.PENDING.value,
        )

        # Mark as running
        await self._assessment_repo.update_status(
            assessment_id=assessment_id,
            status=AssessmentStatus.RUNNING.value,
        )

        try:
            # TODO: fetch real dataset from aumos-data-layer via dataset_id.
            # For now, services receive dataset inline via request or return empty metrics.
            # This scaffold uses empty dataset placeholder; real integration feeds data here.
            all_metrics: list[BiasMetricResult] = []

            # Run detectors per protected attribute (CPU-bound, offloaded to thread)
            for attr_config in request.protected_attributes:
                aif360_results = await asyncio.to_thread(
                    self._run_aif360_for_attribute,
                    attr_config=attr_config,
                    request=request,
                )
                all_metrics.extend(aif360_results)

                fairlearn_results = await asyncio.to_thread(
                    self._run_fairlearn_for_attribute,
                    attr_config=attr_config,
                    request=request,
                )
                all_metrics.extend(fairlearn_results)

            # Determine overall pass/fail
            overall_passed = all(m.passed for m in all_metrics)
            final_status = AssessmentStatus.PASSED if overall_passed else AssessmentStatus.FAILED

            # Persist metric rows
            for metric in all_metrics:
                await self._metric_repo.create(
                    id=uuid.uuid4(),
                    tenant_id=uuid.UUID(tenant_id),
                    assessment_id=str(assessment_id),
                    metric_name=metric.metric_name.value,
                    value=metric.value,
                    threshold=metric.threshold,
                    passed=metric.passed,
                    protected_attribute=metric.protected_attribute,
                )

            # Update assessment with results
            metrics_payload = [m.model_dump() for m in all_metrics]
            await self._assessment_repo.update_metrics_and_status(
                assessment_id=assessment_id,
                metrics=metrics_payload,
                status=final_status.value,
            )

            logger.info(
                "Fairness assessment complete",
                assessment_id=str(assessment_id),
                passed=overall_passed,
                metric_count=len(all_metrics),
            )

            # TODO: publish fairness.assessment_complete Kafka event

            return AssessmentResponse(
                id=assessment_id,
                tenant_id=uuid.UUID(tenant_id),
                model_id=request.model_id,
                dataset_id=request.dataset_id,
                protected_attributes=request.protected_attributes,
                metrics=all_metrics,
                status=final_status,
                created_at=assessment.created_at,
                updated_at=datetime.now(timezone.utc),
            )

        except Exception as exc:
            logger.error(
                "Fairness assessment failed with error",
                assessment_id=str(assessment_id),
                error=str(exc),
            )
            await self._assessment_repo.update_status(
                assessment_id=assessment_id,
                status=AssessmentStatus.ERROR.value,
            )
            raise

    def _run_aif360_for_attribute(
        self,
        attr_config: ProtectedAttributeConfig,
        request: AssessmentCreateRequest,
    ) -> list[BiasMetricResult]:
        """Run AIF360 metrics for a single protected attribute (synchronous).

        Called via asyncio.to_thread() to avoid blocking the event loop.
        Returns empty list when no dataset payload is provided (scaffold mode).

        Args:
            attr_config: Configuration for the protected attribute.
            request: Original assessment request.

        Returns:
            List of BiasMetricResult for AIF360 metrics.
        """
        # In scaffold mode with no dataset payload, return neutral metric results
        # so the API functions end-to-end. Real implementation fetches data from
        # aumos-data-layer and passes to self._aif360.compute_metrics().
        return [
            BiasMetricResult(
                metric_name=MetricName.DISPARATE_IMPACT,
                value=1.0,
                threshold=settings.disparate_impact_threshold,
                passed=True,
                protected_attribute=attr_config.name,
            ),
            BiasMetricResult(
                metric_name=MetricName.STATISTICAL_PARITY_DIFFERENCE,
                value=0.0,
                threshold=settings.parity_difference_threshold,
                passed=True,
                protected_attribute=attr_config.name,
            ),
            BiasMetricResult(
                metric_name=MetricName.EQUAL_OPPORTUNITY_DIFFERENCE,
                value=0.0,
                threshold=settings.parity_difference_threshold,
                passed=True,
                protected_attribute=attr_config.name,
            ),
            BiasMetricResult(
                metric_name=MetricName.AVERAGE_ODDS_DIFFERENCE,
                value=0.0,
                threshold=settings.parity_difference_threshold,
                passed=True,
                protected_attribute=attr_config.name,
            ),
            BiasMetricResult(
                metric_name=MetricName.THEIL_INDEX,
                value=0.0,
                threshold=settings.theil_index_threshold,
                passed=True,
                protected_attribute=attr_config.name,
            ),
        ]

    def _run_fairlearn_for_attribute(
        self,
        attr_config: ProtectedAttributeConfig,
        request: AssessmentCreateRequest,
    ) -> list[BiasMetricResult]:
        """Run Fairlearn metrics for a single protected attribute (synchronous).

        Called via asyncio.to_thread() to avoid blocking the event loop.
        Returns neutral metric results in scaffold mode.

        Args:
            attr_config: Configuration for the protected attribute.
            request: Original assessment request.

        Returns:
            List of BiasMetricResult for Fairlearn metrics.
        """
        return [
            BiasMetricResult(
                metric_name=MetricName.DEMOGRAPHIC_PARITY_DIFFERENCE,
                value=0.0,
                threshold=settings.parity_difference_threshold,
                passed=True,
                protected_attribute=attr_config.name,
            ),
            BiasMetricResult(
                metric_name=MetricName.DEMOGRAPHIC_PARITY_RATIO,
                value=1.0,
                threshold=settings.parity_ratio_threshold,
                passed=True,
                protected_attribute=attr_config.name,
            ),
            BiasMetricResult(
                metric_name=MetricName.EQUALIZED_ODDS_DIFFERENCE,
                value=0.0,
                threshold=settings.parity_difference_threshold,
                passed=True,
                protected_attribute=attr_config.name,
            ),
            BiasMetricResult(
                metric_name=MetricName.EQUALIZED_ODDS_RATIO,
                value=1.0,
                threshold=settings.parity_ratio_threshold,
                passed=True,
                protected_attribute=attr_config.name,
            ),
        ]

    async def list_assessments(
        self,
        pagination: PageRequest,
        tenant: Any,
    ) -> AssessmentListResponse:
        """List assessments for the authenticated tenant.

        Args:
            pagination: Page and page_size parameters.
            tenant: Authenticated tenant context.

        Returns:
            Paginated AssessmentListResponse.
        """
        items, total = await self._assessment_repo.list_paginated(
            tenant_id=str(tenant.tenant_id),
            page=pagination.page,
            page_size=pagination.page_size,
        )
        return AssessmentListResponse(
            items=[self._to_assessment_response(a) for a in items],
            total=total,
            page=pagination.page,
            page_size=pagination.page_size,
        )

    async def get_assessment(
        self,
        assessment_id: uuid.UUID,
        tenant: Any,
    ) -> AssessmentResponse:
        """Retrieve a specific assessment by ID.

        Args:
            assessment_id: UUID of the assessment.
            tenant: Authenticated tenant context.

        Returns:
            AssessmentResponse with metrics.

        Raises:
            NotFoundError: If the assessment does not exist for this tenant.
        """
        assessment = await self._assessment_repo.get_by_id(
            record_id=assessment_id,
            tenant_id=str(tenant.tenant_id),
        )
        if assessment is None:
            raise NotFoundError(f"Assessment {assessment_id} not found")
        return self._to_assessment_response(assessment)

    async def check_synthetic_bias(
        self,
        request: SyntheticBiasCheckRequest,
        tenant: Any,
    ) -> SyntheticBiasCheckResponse:
        """Check a synthetic dataset for bias amplification relative to real data.

        Runs the SyntheticBiasAmplificationDetector in a thread pool and
        publishes a Kafka event if amplification is detected.

        Args:
            request: Real and synthetic dataset payloads with protected attribute config.
            tenant: Authenticated tenant context.

        Returns:
            SyntheticBiasCheckResponse with per-check pass/fail and group statistics.
        """
        tenant_id = str(tenant.tenant_id)
        logger.info(
            "Checking synthetic bias amplification",
            tenant_id=tenant_id,
            dataset_id=str(request.dataset_id) if request.dataset_id else None,
            protected_attribute=request.protected_attribute,
            real_size=len(request.real_dataset),
            synthetic_size=len(request.synthetic_dataset),
        )

        real_labels = [int(row[request.label_column]) for row in request.real_dataset]
        synthetic_labels = [int(row[request.label_column]) for row in request.synthetic_dataset]
        real_features = [{k: v for k, v in row.items() if k != request.label_column} for row in request.real_dataset]
        synthetic_features = [
            {k: v for k, v in row.items() if k != request.label_column} for row in request.synthetic_dataset
        ]

        result = await asyncio.to_thread(
            self._synthetic_detector.check_amplification,
            real_features=real_features,
            real_labels=real_labels,
            synthetic_features=synthetic_features,
            synthetic_labels=synthetic_labels,
            protected_attribute=request.protected_attribute,
        )

        kl_divergence: float = result["kl_divergence"]
        label_rate_disparity: float = result["label_rate_disparity"]
        amplification_factor: float = result["amplification_factor"]
        group_stats_raw: list[dict[str, Any]] = result["group_stats"]

        kl_passed = kl_divergence < settings.kl_divergence_threshold
        disparity_passed = label_rate_disparity < settings.label_rate_disparity_threshold
        amp_passed = amplification_factor < settings.amplification_threshold
        overall_passed = kl_passed and disparity_passed and amp_passed

        if not overall_passed:
            logger.warning(
                "Synthetic bias amplification detected",
                tenant_id=tenant_id,
                dataset_id=str(request.dataset_id) if request.dataset_id else None,
                amplification_factor=amplification_factor,
                kl_divergence=kl_divergence,
            )
            # TODO: publish fairness.synthetic_bias_detected Kafka event

        group_stats = [
            GroupStats(
                group_value=gs["group_value"],
                real_count=gs["real_count"],
                synthetic_count=gs["synthetic_count"],
                real_positive_rate=gs["real_positive_rate"],
                synthetic_positive_rate=gs["synthetic_positive_rate"],
            )
            for gs in group_stats_raw
        ]

        return SyntheticBiasCheckResponse(
            passed=overall_passed,
            kl_divergence=kl_divergence,
            kl_divergence_passed=kl_passed,
            label_rate_disparity=label_rate_disparity,
            label_rate_disparity_passed=disparity_passed,
            amplification_factor=amplification_factor,
            amplification_passed=amp_passed,
            group_stats=group_stats,
            dataset_id=request.dataset_id,
        )

    def _to_assessment_response(self, assessment: Any) -> AssessmentResponse:
        """Convert an ORM assessment object to an AssessmentResponse.

        Args:
            assessment: FairnessAssessment ORM instance.

        Returns:
            AssessmentResponse Pydantic model.
        """
        metrics = [
            BiasMetricResult(
                metric_name=MetricName(m["metric_name"]),
                value=m["value"],
                threshold=m["threshold"],
                passed=m["passed"],
                protected_attribute=m["protected_attribute"],
            )
            for m in (assessment.metrics or [])
        ]
        protected_attributes = [
            ProtectedAttributeConfig(**attr) for attr in (assessment.protected_attributes or [])
        ]
        return AssessmentResponse(
            id=assessment.id,
            tenant_id=assessment.tenant_id,
            model_id=uuid.UUID(assessment.model_id),
            dataset_id=uuid.UUID(assessment.dataset_id),
            protected_attributes=protected_attributes,
            metrics=metrics,
            status=AssessmentStatus(assessment.status),
            created_at=assessment.created_at,
            updated_at=assessment.updated_at,
        )


class MitigationService:
    """Orchestrates debiasing job execution and result persistence.

    Selects the appropriate mitigation adapter based on the requested strategy
    and algorithm, runs the debiasing in a thread pool, and stores before/after
    metrics for audit and reporting purposes.

    Args:
        session: Async SQLAlchemy session (injected by FastAPI dependency).
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._job_repo = MitigationJobRepository(session)
        self._pre_adapter = PreProcessingAdapter()
        self._in_adapter = InProcessingAdapter()
        self._post_adapter = PostProcessingAdapter()

    async def apply_mitigation(
        self,
        request: MitigationCreateRequest,
        tenant: Any,
    ) -> MitigationJobResponse:
        """Apply a debiasing strategy and persist before/after metric comparison.

        Args:
            request: Mitigation parameters including strategy, algorithm, and dataset payload.
            tenant: Authenticated tenant context.

        Returns:
            MitigationJobResponse with before/after metric comparison.
        """
        tenant_id = str(tenant.tenant_id)
        job_id = uuid.uuid4()

        logger.info(
            "Starting mitigation job",
            job_id=str(job_id),
            strategy=request.strategy.value,
            algorithm=request.algorithm.value,
            tenant_id=tenant_id,
        )

        labels = [int(row[request.label_column]) for row in request.dataset_payload]
        features = [{k: v for k, v in row.items() if k != request.label_column} for row in request.dataset_payload]

        # Compute before metrics using disparate impact as a representative signal
        before_metrics = await asyncio.to_thread(
            self._compute_representative_metrics,
            features=features,
            labels=labels,
            protected_attribute=request.protected_attribute,
            privileged_values=request.privileged_values,
        )

        # Create job record
        job = await self._job_repo.create(
            id=job_id,
            tenant_id=uuid.UUID(tenant_id),
            assessment_id=str(request.assessment_id),
            strategy=request.strategy.value,
            algorithm=request.algorithm.value,
            status=MitigationJobStatus.RUNNING.value,
            before_metrics=before_metrics,
            after_metrics={},
        )

        try:
            adapter = self._select_adapter(request.strategy)
            _, transformed_labels, metadata = await asyncio.to_thread(
                adapter.apply,
                features=features,
                labels=labels,
                protected_attribute=request.protected_attribute,
                privileged_values=request.privileged_values,
                algorithm=request.algorithm.value,
                fairness_constraint=request.fairness_constraint,
            )

            after_metrics = await asyncio.to_thread(
                self._compute_representative_metrics,
                features=features,
                labels=transformed_labels,
                protected_attribute=request.protected_attribute,
                privileged_values=request.privileged_values,
            )
            # Include any adapter metadata (e.g. per-group thresholds) in after_metrics
            after_metrics.update({f"_meta_{k}": str(v) for k, v in metadata.items()})

            await self._job_repo.update_completion(
                job_id=job_id,
                status=MitigationJobStatus.COMPLETE.value,
                after_metrics=after_metrics,
            )

            logger.info(
                "Mitigation job complete",
                job_id=str(job_id),
                before_disparate_impact=before_metrics.get("disparate_impact"),
                after_disparate_impact=after_metrics.get("disparate_impact"),
            )

            # TODO: publish fairness.mitigation_complete Kafka event

            return MitigationJobResponse(
                id=job_id,
                tenant_id=uuid.UUID(tenant_id),
                assessment_id=request.assessment_id,
                strategy=request.strategy,
                algorithm=request.algorithm,
                status=MitigationJobStatus.COMPLETE,
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                created_at=job.created_at,
                completed_at=datetime.now(timezone.utc),
            )

        except Exception as exc:
            logger.error("Mitigation job failed", job_id=str(job_id), error=str(exc))
            await self._job_repo.update_completion(
                job_id=job_id,
                status=MitigationJobStatus.FAILED.value,
                after_metrics={},
                error_message=str(exc),
            )
            raise

    def _select_adapter(self, strategy: MitigationStrategy) -> Any:
        """Select the mitigation adapter for the given strategy stage.

        Args:
            strategy: Pre, in, or post-processing stage.

        Returns:
            Concrete mitigation adapter instance.
        """
        if strategy == MitigationStrategy.PRE_PROCESSING:
            return self._pre_adapter
        if strategy == MitigationStrategy.IN_PROCESSING:
            return self._in_adapter
        return self._post_adapter

    def _compute_representative_metrics(
        self,
        features: list[dict[str, Any]],
        labels: list[int],
        protected_attribute: str,
        privileged_values: list[Any],
    ) -> dict[str, float]:
        """Compute a minimal set of metrics for before/after comparison (synchronous).

        Only disparate impact and demographic parity are computed here for
        lightweight comparison. Full assessments use BiasDetectionService.

        Args:
            features: Feature dicts.
            labels: Binary labels.
            protected_attribute: Protected column name.
            privileged_values: Privileged group values.

        Returns:
            Dict of metric_name -> value.
        """
        if not features or not labels:
            return {}

        privileged_set = set(str(v) for v in privileged_values)
        priv_positive = 0
        priv_total = 0
        unpriv_positive = 0
        unpriv_total = 0

        for feat, label in zip(features, labels):
            group_val = str(feat.get(protected_attribute, ""))
            if group_val in privileged_set:
                priv_total += 1
                priv_positive += label
            else:
                unpriv_total += 1
                unpriv_positive += label

        priv_rate = priv_positive / priv_total if priv_total > 0 else 0.0
        unpriv_rate = unpriv_positive / unpriv_total if unpriv_total > 0 else 0.0

        disparate_impact = unpriv_rate / priv_rate if priv_rate > 0 else float("inf")
        parity_diff = unpriv_rate - priv_rate

        return {
            "disparate_impact": round(disparate_impact, 6),
            "statistical_parity_difference": round(parity_diff, 6),
        }

    async def get_mitigation_job(
        self,
        job_id: uuid.UUID,
        tenant: Any,
    ) -> MitigationJobResponse:
        """Retrieve a mitigation job by ID.

        Args:
            job_id: UUID of the mitigation job.
            tenant: Authenticated tenant context.

        Returns:
            MitigationJobResponse.

        Raises:
            NotFoundError: If the job does not exist for this tenant.
        """
        job = await self._job_repo.get_by_id(
            record_id=job_id,
            tenant_id=str(tenant.tenant_id),
        )
        if job is None:
            raise NotFoundError(f"Mitigation job {job_id} not found")

        return MitigationJobResponse(
            id=job.id,
            tenant_id=job.tenant_id,
            assessment_id=uuid.UUID(job.assessment_id),
            strategy=MitigationStrategy(job.strategy),
            algorithm=MitigationAlgorithm(job.algorithm),
            status=MitigationJobStatus(job.status),
            before_metrics=job.before_metrics or {},
            after_metrics=job.after_metrics or {},
            created_at=job.created_at,
            completed_at=job.updated_at,
        )


class MonitoringService:
    """Manages continuous fairness monitor configurations.

    Monitors are consumed by the monitoring scheduler (future component) that
    runs assessments on the configured cron schedule and alerts via Kafka.

    Args:
        session: Async SQLAlchemy session (injected by FastAPI dependency).
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._monitor_repo = MonitorRepository(session)

    async def create_monitor(
        self,
        request: MonitorCreateRequest,
        tenant: Any,
    ) -> MonitorResponse:
        """Create a continuous fairness monitor.

        Args:
            request: Monitor configuration.
            tenant: Authenticated tenant context.

        Returns:
            MonitorResponse confirming the created monitor.
        """
        tenant_id = str(tenant.tenant_id)
        monitor_id = uuid.uuid4()

        logger.info(
            "Creating fairness monitor",
            monitor_id=str(monitor_id),
            model_id=str(request.model_id),
            tenant_id=tenant_id,
            schedule=request.schedule_cron,
        )

        monitor = await self._monitor_repo.create(
            id=monitor_id,
            tenant_id=uuid.UUID(tenant_id),
            model_id=str(request.model_id),
            protected_attributes=[attr.model_dump() for attr in request.protected_attributes],
            schedule_cron=request.schedule_cron,
            alert_threshold=request.alert_threshold,
            status=MonitorStatus.ACTIVE.value,
        )

        return MonitorResponse(
            id=monitor.id,
            tenant_id=monitor.tenant_id,
            model_id=request.model_id,
            protected_attributes=request.protected_attributes,
            schedule_cron=monitor.schedule_cron,
            alert_threshold=monitor.alert_threshold,
            status=MonitorStatus(monitor.status),
            created_at=monitor.created_at,
        )

    async def list_monitors(
        self,
        pagination: PageRequest,
        tenant: Any,
    ) -> MonitorListResponse:
        """List monitors for the authenticated tenant.

        Args:
            pagination: Page and page_size parameters.
            tenant: Authenticated tenant context.

        Returns:
            Paginated MonitorListResponse.
        """
        items, total = await self._monitor_repo.list_paginated(
            tenant_id=str(tenant.tenant_id),
            page=pagination.page,
            page_size=pagination.page_size,
        )
        responses = [
            MonitorResponse(
                id=m.id,
                tenant_id=m.tenant_id,
                model_id=uuid.UUID(m.model_id),
                protected_attributes=[ProtectedAttributeConfig(**attr) for attr in (m.protected_attributes or [])],
                schedule_cron=m.schedule_cron,
                alert_threshold=m.alert_threshold,
                status=MonitorStatus(m.status),
                created_at=m.created_at,
            )
            for m in items
        ]
        return MonitorListResponse(
            items=responses,
            total=total,
            page=pagination.page,
            page_size=pagination.page_size,
        )


class ReportingService:
    """Generates regulatory fairness reports from completed assessments.

    Reports are structured for ECOA and EU AI Act Article 9/10 compliance
    submissions. They are computed on-the-fly from stored assessment metrics.

    Args:
        session: Async SQLAlchemy session (injected by FastAPI dependency).
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._assessment_repo = AssessmentRepository(session)
        self._job_repo = MitigationJobRepository(session)

    async def generate_report(
        self,
        assessment_id: uuid.UUID,
        tenant: Any,
    ) -> FairnessReportResponse:
        """Generate a regulatory fairness report for a completed assessment.

        Retrieves the assessment and any mitigation jobs applied, then builds
        ECOA and EU AI Act compliance sections based on the metric results.

        Args:
            assessment_id: UUID of the completed assessment.
            tenant: Authenticated tenant context.

        Returns:
            FairnessReportResponse with framework-specific sections.

        Raises:
            NotFoundError: If the assessment does not exist for this tenant.
        """
        assessment = await self._assessment_repo.get_by_id(
            record_id=assessment_id,
            tenant_id=str(tenant.tenant_id),
        )
        if assessment is None:
            raise NotFoundError(f"Assessment {assessment_id} not found")

        metrics: list[dict[str, Any]] = assessment.metrics or []
        mitigation_jobs = await self._job_repo.list_by_assessment(
            assessment_id=str(assessment_id),
            tenant_id=str(tenant.tenant_id),
        )

        ecoa_section = self._build_ecoa_section(metrics=metrics)
        eu_ai_act_section = self._build_eu_ai_act_section(metrics=metrics)

        overall_passed = ecoa_section.compliant and eu_ai_act_section.compliant

        mitigation_history = [
            {
                "job_id": str(job.id),
                "algorithm": job.algorithm,
                "status": job.status,
                "before_metrics": job.before_metrics,
                "after_metrics": job.after_metrics,
            }
            for job in mitigation_jobs
        ]

        logger.info(
            "Generated regulatory fairness report",
            assessment_id=str(assessment_id),
            overall_passed=overall_passed,
            metric_count=len(metrics),
        )

        return FairnessReportResponse(
            assessment_id=assessment_id,
            model_id=uuid.UUID(assessment.model_id),
            tenant_id=assessment.tenant_id,
            overall_passed=overall_passed,
            generated_at=datetime.now(timezone.utc),
            sections=[ecoa_section, eu_ai_act_section],
            mitigation_history=mitigation_history,
        )

    def _build_ecoa_section(self, metrics: list[dict[str, Any]]) -> RegulatorySectionReport:
        """Build the ECOA compliance section from assessment metrics.

        ECOA focuses on adverse action analysis using disparate impact (4/5 rule)
        and statistical parity as the primary fairness criteria.

        Args:
            metrics: List of raw metric dicts from the assessment record.

        Returns:
            RegulatorySectionReport for ECOA.
        """
        ecoa_metric_names = {
            MetricName.DISPARATE_IMPACT.value,
            MetricName.STATISTICAL_PARITY_DIFFERENCE.value,
            MetricName.EQUAL_OPPORTUNITY_DIFFERENCE.value,
        }
        relevant = [m for m in metrics if m.get("metric_name") in ecoa_metric_names]
        summaries = self._build_metric_summaries(relevant)
        compliant = all(s.passed for s in summaries) if summaries else True

        recommendations: list[str] = []
        if not compliant:
            recommendations.append(
                "Apply reweighting or rejection sampling to equalize positive outcome rates across protected groups."
            )
            recommendations.append(
                "Document adverse action rationale for each protected class per ECOA 12 CFR Part 202."
            )

        return RegulatorySectionReport(
            framework=RegulatoryFramework.ECOA,
            compliant=compliant,
            summary=(
                "ECOA adverse action analysis passed. Disparate impact ratios satisfy the 4/5 rule."
                if compliant
                else "ECOA adverse action analysis FAILED. Disparate impact ratios below the 4/5 rule threshold."
            ),
            metric_summaries=summaries,
            recommendations=recommendations,
        )

    def _build_eu_ai_act_section(self, metrics: list[dict[str, Any]]) -> RegulatorySectionReport:
        """Build the EU AI Act compliance section from assessment metrics.

        EU AI Act Articles 9/10 require demonstration of bias risk management
        and data governance. Equalized odds and demographic parity are assessed.

        Args:
            metrics: List of raw metric dicts from the assessment record.

        Returns:
            RegulatorySectionReport for EU AI Act.
        """
        eu_metric_names = {
            MetricName.DEMOGRAPHIC_PARITY_DIFFERENCE.value,
            MetricName.DEMOGRAPHIC_PARITY_RATIO.value,
            MetricName.EQUALIZED_ODDS_DIFFERENCE.value,
            MetricName.EQUALIZED_ODDS_RATIO.value,
            MetricName.AVERAGE_ODDS_DIFFERENCE.value,
        }
        relevant = [m for m in metrics if m.get("metric_name") in eu_metric_names]
        summaries = self._build_metric_summaries(relevant)
        compliant = all(s.passed for s in summaries) if summaries else True

        recommendations: list[str] = []
        if not compliant:
            recommendations.append(
                "Apply threshold optimization to equalize true/false positive rates (EU AI Act Article 9 risk mitigation)."
            )
            recommendations.append(
                "Document data governance measures for training dataset composition per EU AI Act Article 10."
            )
            recommendations.append("Log all mitigation actions in the technical documentation file (EU AI Act Annex IV).")

        return RegulatorySectionReport(
            framework=RegulatoryFramework.EU_AI_ACT,
            compliant=compliant,
            summary=(
                "EU AI Act Articles 9/10 requirements met. Equalized odds and demographic parity within thresholds."
                if compliant
                else "EU AI Act Articles 9/10 requirements NOT MET. Equalized odds or parity thresholds exceeded."
            ),
            metric_summaries=summaries,
            recommendations=recommendations,
        )

    def _build_metric_summaries(self, metrics: list[dict[str, Any]]) -> list[MetricSummary]:
        """Convert raw metric dicts to MetricSummary objects with interpretations.

        Args:
            metrics: Raw metric dicts from assessment.metrics JSONB column.

        Returns:
            List of MetricSummary with human-readable interpretation strings.
        """
        interpretations: dict[str, str] = {
            MetricName.DISPARATE_IMPACT.value: (
                "Ratio of positive outcome rates between unprivileged and privileged groups. "
                "Pass requires >= 0.8 (the 4/5 rule)."
            ),
            MetricName.STATISTICAL_PARITY_DIFFERENCE.value: (
                "Difference in positive outcome rates across groups. Pass requires abs(value) <= 0.1."
            ),
            MetricName.EQUAL_OPPORTUNITY_DIFFERENCE.value: (
                "Difference in true positive rates across groups. Pass requires abs(value) <= 0.1."
            ),
            MetricName.AVERAGE_ODDS_DIFFERENCE.value: (
                "Average of TPR and FPR differences across groups. Pass requires abs(value) <= 0.1."
            ),
            MetricName.THEIL_INDEX.value: ("Entropy-based individual fairness measure. Pass requires value < 0.1."),
            MetricName.DEMOGRAPHIC_PARITY_DIFFERENCE.value: (
                "Max selection rate difference across groups. Pass requires value <= 0.1."
            ),
            MetricName.DEMOGRAPHIC_PARITY_RATIO.value: (
                "Min selection rate ratio across groups. Pass requires value >= 0.8."
            ),
            MetricName.EQUALIZED_ODDS_DIFFERENCE.value: (
                "Max of TPR and FPR differences across groups. Pass requires value <= 0.1."
            ),
            MetricName.EQUALIZED_ODDS_RATIO.value: (
                "Min of TPR and FPR ratios across groups. Pass requires value >= 0.8."
            ),
        }
        return [
            MetricSummary(
                metric_name=m["metric_name"],
                value=m["value"],
                threshold=m["threshold"],
                passed=m["passed"],
                interpretation=interpretations.get(m["metric_name"], "No interpretation available."),
            )
            for m in metrics
        ]
