"""FastAPI router for the Fairness Suite API.

All routes are thin: they validate the request, extract auth context, delegate
to the appropriate service, and return the response schema. No business logic here.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.auth import get_current_tenant
from aumos_common.database import get_db_session
from aumos_common.pagination import PageRequest

from aumos_fairness_suite.api.schemas import (
    AssessmentCreateRequest,
    AssessmentListResponse,
    AssessmentResponse,
    FairnessReportResponse,
    MitigationCreateRequest,
    MitigationJobResponse,
    MonitorCreateRequest,
    MonitorListResponse,
    MonitorResponse,
    SyntheticBiasCheckRequest,
    SyntheticBiasCheckResponse,
)
from aumos_fairness_suite.core.services import (
    BiasDetectionService,
    MitigationService,
    MonitoringService,
    ReportingService,
)

router = APIRouter(tags=["fairness"])


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


def get_bias_detection_service(
    session: AsyncSession = Depends(get_db_session),
) -> BiasDetectionService:
    """Construct BiasDetectionService with injected database session."""
    return BiasDetectionService(session=session)


def get_mitigation_service(
    session: AsyncSession = Depends(get_db_session),
) -> MitigationService:
    """Construct MitigationService with injected database session."""
    return MitigationService(session=session)


def get_monitoring_service(
    session: AsyncSession = Depends(get_db_session),
) -> MonitoringService:
    """Construct MonitoringService with injected database session."""
    return MonitoringService(session=session)


def get_reporting_service(
    session: AsyncSession = Depends(get_db_session),
) -> ReportingService:
    """Construct ReportingService with injected database session."""
    return ReportingService(session=session)


# ---------------------------------------------------------------------------
# Assessment endpoints
# ---------------------------------------------------------------------------


@router.post("/assessments", response_model=AssessmentResponse, status_code=201)
async def create_assessment(
    request: AssessmentCreateRequest,
    tenant: object = Depends(get_current_tenant),
    service: BiasDetectionService = Depends(get_bias_detection_service),
) -> AssessmentResponse:
    """Run a fairness assessment on a model and dataset.

    Computes AIF360 and Fairlearn bias metrics for each protected attribute
    specified in the request and returns the assessment with pass/fail results.

    Args:
        request: Assessment parameters including model ID, dataset, and protected attributes.
        tenant: Authenticated tenant context (injected by aumos-common).
        service: BiasDetectionService (injected).

    Returns:
        AssessmentResponse with computed metrics and overall status.
    """
    return await service.create_assessment(request=request, tenant=tenant)


@router.get("/assessments", response_model=AssessmentListResponse)
async def list_assessments(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    tenant: object = Depends(get_current_tenant),
    service: BiasDetectionService = Depends(get_bias_detection_service),
) -> AssessmentListResponse:
    """List fairness assessments for the authenticated tenant (paginated).

    Args:
        page: Page number (1-indexed).
        page_size: Number of items per page.
        tenant: Authenticated tenant context (injected).
        service: BiasDetectionService (injected).

    Returns:
        Paginated list of AssessmentResponse objects.
    """
    pagination = PageRequest(page=page, page_size=page_size)
    return await service.list_assessments(pagination=pagination, tenant=tenant)


@router.get("/assessments/{assessment_id}", response_model=AssessmentResponse)
async def get_assessment(
    assessment_id: uuid.UUID,
    tenant: object = Depends(get_current_tenant),
    service: BiasDetectionService = Depends(get_bias_detection_service),
) -> AssessmentResponse:
    """Get a specific fairness assessment including all computed bias metrics.

    Args:
        assessment_id: UUID of the assessment to retrieve.
        tenant: Authenticated tenant context (injected).
        service: BiasDetectionService (injected).

    Returns:
        AssessmentResponse with full metric details.

    Raises:
        NotFoundError: If the assessment does not exist or belongs to another tenant.
    """
    return await service.get_assessment(assessment_id=assessment_id, tenant=tenant)


# ---------------------------------------------------------------------------
# Mitigation endpoints
# ---------------------------------------------------------------------------


@router.post("/mitigations", response_model=MitigationJobResponse, status_code=201)
async def create_mitigation(
    request: MitigationCreateRequest,
    tenant: object = Depends(get_current_tenant),
    service: MitigationService = Depends(get_mitigation_service),
) -> MitigationJobResponse:
    """Apply a debiasing strategy to reduce bias identified in an assessment.

    Supports pre-processing (reweighting, rejection sampling), in-processing
    (adversarial debiasing), and post-processing (threshold optimization) strategies.

    Args:
        request: Mitigation parameters including strategy, algorithm, and dataset payload.
        tenant: Authenticated tenant context (injected).
        service: MitigationService (injected).

    Returns:
        MitigationJobResponse with before/after metrics once the job completes.
    """
    return await service.apply_mitigation(request=request, tenant=tenant)


@router.get("/mitigations/{job_id}", response_model=MitigationJobResponse)
async def get_mitigation(
    job_id: uuid.UUID,
    tenant: object = Depends(get_current_tenant),
    service: MitigationService = Depends(get_mitigation_service),
) -> MitigationJobResponse:
    """Get the results of a mitigation job.

    Args:
        job_id: UUID of the mitigation job to retrieve.
        tenant: Authenticated tenant context (injected).
        service: MitigationService (injected).

    Returns:
        MitigationJobResponse with status and metric comparison.

    Raises:
        NotFoundError: If the job does not exist or belongs to another tenant.
    """
    return await service.get_mitigation_job(job_id=job_id, tenant=tenant)


# ---------------------------------------------------------------------------
# Monitor endpoints
# ---------------------------------------------------------------------------


@router.post("/monitors", response_model=MonitorResponse, status_code=201)
async def create_monitor(
    request: MonitorCreateRequest,
    tenant: object = Depends(get_current_tenant),
    service: MonitoringService = Depends(get_monitoring_service),
) -> MonitorResponse:
    """Create a continuous fairness monitor for a deployed model.

    Monitors run on a cron schedule and emit `fairness.monitor_alert` Kafka events
    when any metric exceeds the configured alert threshold.

    Args:
        request: Monitor configuration including model ID, schedule, and threshold.
        tenant: Authenticated tenant context (injected).
        service: MonitoringService (injected).

    Returns:
        MonitorResponse confirming the created monitor configuration.
    """
    return await service.create_monitor(request=request, tenant=tenant)


@router.get("/monitors", response_model=MonitorListResponse)
async def list_monitors(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    tenant: object = Depends(get_current_tenant),
    service: MonitoringService = Depends(get_monitoring_service),
) -> MonitorListResponse:
    """List fairness monitors for the authenticated tenant (paginated).

    Args:
        page: Page number (1-indexed).
        page_size: Number of items per page.
        tenant: Authenticated tenant context (injected).
        service: MonitoringService (injected).

    Returns:
        Paginated list of MonitorResponse objects.
    """
    pagination = PageRequest(page=page, page_size=page_size)
    return await service.list_monitors(pagination=pagination, tenant=tenant)


# ---------------------------------------------------------------------------
# Synthetic bias check endpoint
# ---------------------------------------------------------------------------


@router.post("/synthetic-bias-check", response_model=SyntheticBiasCheckResponse)
async def check_synthetic_bias(
    request: SyntheticBiasCheckRequest,
    tenant: object = Depends(get_current_tenant),
    service: BiasDetectionService = Depends(get_bias_detection_service),
) -> SyntheticBiasCheckResponse:
    """Check whether a synthetic dataset amplifies bias present in the real dataset.

    Computes KL divergence of group distributions, per-group label rate disparity,
    and the amplification factor (bias in synthetic / bias in real). Emits a
    `fairness.synthetic_bias_detected` Kafka event when amplification is detected.

    Args:
        request: Real and synthetic dataset payloads with protected attribute config.
        tenant: Authenticated tenant context (injected).
        service: BiasDetectionService (injected).

    Returns:
        SyntheticBiasCheckResponse with pass/fail per check and aggregate statistics.
    """
    return await service.check_synthetic_bias(request=request, tenant=tenant)


# ---------------------------------------------------------------------------
# Report endpoint
# ---------------------------------------------------------------------------


@router.get("/reports/{assessment_id}", response_model=FairnessReportResponse)
async def get_fairness_report(
    assessment_id: uuid.UUID,
    tenant: object = Depends(get_current_tenant),
    service: ReportingService = Depends(get_reporting_service),
) -> FairnessReportResponse:
    """Generate a regulatory fairness report for a completed assessment.

    The report covers ECOA (Equal Credit Opportunity Act) and EU AI Act Article 9/10
    requirements and includes metric summaries, pass/fail verdicts, and recommendations.

    Args:
        assessment_id: UUID of the completed assessment to report on.
        tenant: Authenticated tenant context (injected).
        service: ReportingService (injected).

    Returns:
        FairnessReportResponse with regulatory section breakdown.

    Raises:
        NotFoundError: If the assessment does not exist or belongs to another tenant.
    """
    return await service.generate_report(assessment_id=assessment_id, tenant=tenant)
