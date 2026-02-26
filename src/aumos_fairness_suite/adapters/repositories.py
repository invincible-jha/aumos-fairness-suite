"""SQLAlchemy repository implementations for the Fairness Suite.

All repositories extend BaseRepository from aumos-common, which provides
RLS-enforced query helpers, get_by_id, create, and update primitives.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.database import BaseRepository
from aumos_common.observability import get_logger

from aumos_fairness_suite.core.models import (
    BiasMetric,
    FairnessAssessment,
    FairnessMonitor,
    MitigationJob,
)

logger = get_logger(__name__)


class AssessmentRepository(BaseRepository[FairnessAssessment]):
    """Repository for FairnessAssessment (fai_assessments table).

    Args:
        session: Async SQLAlchemy session.
    """

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session=session, model=FairnessAssessment)

    async def update_status(
        self,
        assessment_id: uuid.UUID,
        status: str,
    ) -> None:
        """Update the status of an assessment.

        Args:
            assessment_id: UUID of the assessment to update.
            status: New status string.
        """
        stmt = (
            update(FairnessAssessment)
            .where(FairnessAssessment.id == assessment_id)
            .values(status=status)
        )
        await self._session.execute(stmt)
        await self._session.flush()

    async def update_metrics_and_status(
        self,
        assessment_id: uuid.UUID,
        metrics: list[dict[str, Any]],
        status: str,
    ) -> None:
        """Update the metrics JSONB column and status of an assessment.

        Args:
            assessment_id: UUID of the assessment to update.
            metrics: Serialised list of metric result dicts.
            status: Final lifecycle status.
        """
        stmt = (
            update(FairnessAssessment)
            .where(FairnessAssessment.id == assessment_id)
            .values(metrics=metrics, status=status)
        )
        await self._session.execute(stmt)
        await self._session.flush()

    async def list_paginated(
        self,
        tenant_id: str,
        page: int,
        page_size: int,
    ) -> tuple[list[FairnessAssessment], int]:
        """List assessments for a tenant with pagination.

        Args:
            tenant_id: Tenant UUID string to filter by.
            page: Page number (1-indexed).
            page_size: Number of items per page.

        Returns:
            Tuple of (list of assessments, total count).
        """
        offset = (page - 1) * page_size
        stmt = (
            select(FairnessAssessment)
            .where(FairnessAssessment.tenant_id == uuid.UUID(tenant_id))
            .order_by(FairnessAssessment.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        result = await self._session.execute(stmt)
        items = list(result.scalars().all())

        count_stmt = (
            select(FairnessAssessment)
            .where(FairnessAssessment.tenant_id == uuid.UUID(tenant_id))
        )
        count_result = await self._session.execute(count_stmt)
        total = len(count_result.scalars().all())

        return items, total

    async def get_by_id(
        self,
        record_id: uuid.UUID,
        tenant_id: str,
    ) -> FairnessAssessment | None:
        """Get an assessment by ID, scoped to the tenant.

        Args:
            record_id: Assessment UUID.
            tenant_id: Tenant UUID string for RLS enforcement.

        Returns:
            FairnessAssessment ORM instance or None if not found.
        """
        stmt = select(FairnessAssessment).where(
            FairnessAssessment.id == record_id,
            FairnessAssessment.tenant_id == uuid.UUID(tenant_id),
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()


class BiasMetricRepository(BaseRepository[BiasMetric]):
    """Repository for BiasMetric (fai_bias_metrics table).

    Args:
        session: Async SQLAlchemy session.
    """

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session=session, model=BiasMetric)

    async def list_by_assessment(
        self,
        assessment_id: str,
        tenant_id: str,
    ) -> list[BiasMetric]:
        """List all metric rows for a given assessment.

        Args:
            assessment_id: Assessment UUID string.
            tenant_id: Tenant UUID string for RLS enforcement.

        Returns:
            List of BiasMetric ORM instances.
        """
        stmt = select(BiasMetric).where(
            BiasMetric.assessment_id == assessment_id,
            BiasMetric.tenant_id == uuid.UUID(tenant_id),
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())


class MitigationJobRepository(BaseRepository[MitigationJob]):
    """Repository for MitigationJob (fai_mitigation_jobs table).

    Args:
        session: Async SQLAlchemy session.
    """

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session=session, model=MitigationJob)

    async def update_completion(
        self,
        job_id: uuid.UUID,
        status: str,
        after_metrics: dict[str, Any],
        error_message: str | None = None,
    ) -> None:
        """Update a job's status, after_metrics, and optional error message.

        Args:
            job_id: UUID of the mitigation job.
            status: Final job status.
            after_metrics: Metrics computed after mitigation.
            error_message: Error description if status is "failed".
        """
        values: dict[str, Any] = {"status": status, "after_metrics": after_metrics}
        if error_message is not None:
            values["error_message"] = error_message
        stmt = update(MitigationJob).where(MitigationJob.id == job_id).values(**values)
        await self._session.execute(stmt)
        await self._session.flush()

    async def list_by_assessment(
        self,
        assessment_id: str,
        tenant_id: str,
    ) -> list[MitigationJob]:
        """List all mitigation jobs for a given assessment.

        Args:
            assessment_id: Assessment UUID string.
            tenant_id: Tenant UUID string for RLS enforcement.

        Returns:
            List of MitigationJob ORM instances.
        """
        stmt = select(MitigationJob).where(
            MitigationJob.assessment_id == assessment_id,
            MitigationJob.tenant_id == uuid.UUID(tenant_id),
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_id(
        self,
        record_id: uuid.UUID,
        tenant_id: str,
    ) -> MitigationJob | None:
        """Get a mitigation job by ID, scoped to the tenant.

        Args:
            record_id: Job UUID.
            tenant_id: Tenant UUID string for RLS enforcement.

        Returns:
            MitigationJob ORM instance or None.
        """
        stmt = select(MitigationJob).where(
            MitigationJob.id == record_id,
            MitigationJob.tenant_id == uuid.UUID(tenant_id),
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()


class MonitorRepository(BaseRepository[FairnessMonitor]):
    """Repository for FairnessMonitor (fai_monitors table).

    Args:
        session: Async SQLAlchemy session.
    """

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session=session, model=FairnessMonitor)

    async def list_paginated(
        self,
        tenant_id: str,
        page: int,
        page_size: int,
    ) -> tuple[list[FairnessMonitor], int]:
        """List monitors for a tenant with pagination.

        Args:
            tenant_id: Tenant UUID string.
            page: Page number (1-indexed).
            page_size: Number of items per page.

        Returns:
            Tuple of (list of monitors, total count).
        """
        offset = (page - 1) * page_size
        stmt = (
            select(FairnessMonitor)
            .where(FairnessMonitor.tenant_id == uuid.UUID(tenant_id))
            .order_by(FairnessMonitor.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        result = await self._session.execute(stmt)
        items = list(result.scalars().all())

        count_stmt = select(FairnessMonitor).where(FairnessMonitor.tenant_id == uuid.UUID(tenant_id))
        count_result = await self._session.execute(count_stmt)
        total = len(count_result.scalars().all())

        return items, total
