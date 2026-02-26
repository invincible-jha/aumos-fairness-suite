"""Kafka event publisher for fairness lifecycle events.

Publishes domain events to Kafka topics after state changes in the Fairness Suite.
Uses the EventPublisher from aumos-common with standardised topic names.
"""

from __future__ import annotations

from aumos_common.events import EventPublisher
from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Fairness-specific Kafka topic names
TOPIC_ASSESSMENT_COMPLETE = "fairness.assessment_complete"
TOPIC_ASSESSMENT_FAILED = "fairness.assessment_failed"
TOPIC_MITIGATION_COMPLETE = "fairness.mitigation_complete"
TOPIC_MONITOR_ALERT = "fairness.monitor_alert"
TOPIC_SYNTHETIC_BIAS_DETECTED = "fairness.synthetic_bias_detected"


class FairnessEventPublisher:
    """Publishes fairness lifecycle events to Kafka.

    Wraps aumos-common EventPublisher with fairness-specific topic constants
    and structured event payloads.

    Args:
        publisher: aumos-common EventPublisher instance (injected).
    """

    def __init__(self, publisher: EventPublisher) -> None:
        self._publisher = publisher

    async def publish_assessment_complete(
        self,
        tenant_id: str,
        assessment_id: str,
        model_id: str,
        passed: bool,
    ) -> None:
        """Publish fairness.assessment_complete or fairness.assessment_failed event.

        Sends to the appropriate topic based on whether all metrics passed,
        so downstream consumers can filter by topic rather than payload.

        Args:
            tenant_id: Owning tenant UUID string.
            assessment_id: Assessment UUID string.
            model_id: Model UUID string.
            passed: True if all metrics passed, False if any failed.
        """
        topic = TOPIC_ASSESSMENT_COMPLETE if passed else TOPIC_ASSESSMENT_FAILED
        payload = {
            "tenant_id": tenant_id,
            "assessment_id": assessment_id,
            "model_id": model_id,
            "passed": passed,
        }
        await self._publisher.publish(topic=topic, payload=payload)
        logger.info(
            "Published fairness assessment event",
            topic=topic,
            assessment_id=assessment_id,
            passed=passed,
        )

    async def publish_mitigation_complete(
        self,
        tenant_id: str,
        job_id: str,
        assessment_id: str,
        algorithm: str,
    ) -> None:
        """Publish fairness.mitigation_complete event.

        Args:
            tenant_id: Owning tenant UUID string.
            job_id: Mitigation job UUID string.
            assessment_id: Linked assessment UUID string.
            algorithm: Algorithm name applied.
        """
        payload = {
            "tenant_id": tenant_id,
            "job_id": job_id,
            "assessment_id": assessment_id,
            "algorithm": algorithm,
        }
        await self._publisher.publish(topic=TOPIC_MITIGATION_COMPLETE, payload=payload)
        logger.info("Published mitigation complete event", job_id=job_id, algorithm=algorithm)

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
        payload = {
            "tenant_id": tenant_id,
            "monitor_id": monitor_id,
            "model_id": model_id,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "threshold": threshold,
        }
        await self._publisher.publish(topic=TOPIC_MONITOR_ALERT, payload=payload)
        logger.warning(
            "Published monitor alert",
            monitor_id=monitor_id,
            model_id=model_id,
            metric_name=metric_name,
            metric_value=metric_value,
        )

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
        payload = {
            "tenant_id": tenant_id,
            "dataset_id": dataset_id,
            "amplification_factor": amplification_factor,
        }
        await self._publisher.publish(topic=TOPIC_SYNTHETIC_BIAS_DETECTED, payload=payload)
        logger.warning(
            "Published synthetic bias detected event",
            dataset_id=dataset_id,
            amplification_factor=amplification_factor,
        )
