"""AumOS Fairness Suite service entry point."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from aumos_common.app import create_app
from aumos_common.database import init_database
from aumos_common.observability import get_logger

from aumos_fairness_suite.settings import Settings

logger = get_logger(__name__)
settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage startup and shutdown lifecycle.

    Initialises the database connection pool and Kafka publisher on startup,
    then cleanly shuts down on service termination.
    """
    logger.info("Starting aumos-fairness-suite", version="0.1.0")
    init_database(settings.database)
    # TODO: initialise Kafka publisher for fairness events
    yield
    logger.info("Shutting down aumos-fairness-suite")


app: FastAPI = create_app(
    service_name="aumos-fairness-suite",
    version="0.1.0",
    settings=settings,
    lifespan=lifespan,
    health_checks=[
        # HealthCheck(name="postgres", check_fn=check_db),
    ],
)

from aumos_fairness_suite.api.router import router  # noqa: E402

app.include_router(router, prefix="/api/v1")
