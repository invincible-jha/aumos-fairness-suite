# CLAUDE.md — AumOS Fairness Suite

## Project Overview

AumOS Enterprise is a composable enterprise AI platform with 9 products + 2 services
across 62 repositories. This repo (`aumos-fairness-suite`) is part of **Tier 3: AI
Governance and Compliance**: bias detection, mitigation, and continuous fairness
monitoring for deployed ML models.

**Release Tier:** B: Open Core
**Product Mapping:** Product 7 — AI Governance Engine
**Phase:** 3B (Months 14-20)

## Repo Purpose

`aumos-fairness-suite` provides end-to-end bias detection and mitigation for ML models
deployed within the AumOS platform. It combines IBM AI Fairness 360 and Microsoft
Fairlearn metrics with pre-, in-, and post-processing debiasing strategies. Continuous
monitoring with cron-driven schedules detects emerging bias in production. A synthetic
data bias amplification detector identifies when generative models encode and amplify
demographic biases from training data.

## Architecture Position

```
aumos-common        → base models, auth, database, events, observability
aumos-proto         → Protobuf schemas for fairness events
aumos-governance-engine → orchestrates assessments, receives reports

aumos-fairness-suite (THIS REPO)
    ↓
aumos-governance-engine  → receives fairness.assessment_complete events
aumos-mlops-lifecycle    → triggers model retraining after failed assessments
aumos-observability      → receives fairness metrics for dashboards
```

**Upstream dependencies (this repo IMPORTS from):**
- `aumos-common` — auth, database, events, errors, config, health, pagination
- `aumos-proto` — Protobuf message definitions for Kafka events
- `aumos-governance-engine` — assessment orchestration requests

**Downstream dependents (other repos IMPORT from this):**
- `aumos-governance-engine` — receives fairness assessment results and regulatory reports
- `aumos-mlops-lifecycle` — receives `fairness.failed` events to trigger retraining
- `aumos-observability` — receives metric timeseries for fairness dashboards

## Tech Stack (DO NOT DEVIATE)

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11+ | Runtime |
| FastAPI | 0.110+ | REST API framework |
| SQLAlchemy | 2.0+ (async) | Database ORM |
| asyncpg | 0.29+ | PostgreSQL async driver |
| Pydantic | 2.6+ | Data validation, settings, API schemas |
| confluent-kafka | 2.3+ | Kafka producer/consumer |
| structlog | 24.1+ | Structured JSON logging |
| OpenTelemetry | 1.23+ | Distributed tracing |
| AI Fairness 360 | 0.6.1+ | IBM bias metrics (disparate impact, equal opportunity) |
| Fairlearn | 0.10+ | Microsoft fairness metrics (demographic parity, equalized odds) |
| scikit-learn | 1.4+ | ML model interfaces for in-processing debiasing |
| scipy | 1.12+ | Statistical tests for synthetic bias detection |
| pandas | 2.1+ | DataFrame operations for dataset analysis |
| pytest | 8.0+ | Testing framework |
| ruff | 0.3+ | Linting and formatting |
| mypy | 1.8+ | Type checking |

## Coding Standards

### ABSOLUTE RULES (violations will break integration with other repos)

1. **Import aumos-common, never reimplement.** If aumos-common provides it, use it.
   ```python
   # CORRECT
   from aumos_common.auth import get_current_tenant, get_current_user
   from aumos_common.database import get_db_session, Base, AumOSModel, BaseRepository
   from aumos_common.events import EventPublisher, Topics
   from aumos_common.errors import NotFoundError, ErrorCode
   from aumos_common.config import AumOSSettings
   from aumos_common.health import create_health_router
   from aumos_common.pagination import PageRequest, PageResponse, paginate
   from aumos_common.app import create_app

   # WRONG — never reimplement these
   # from jose import jwt
   # from sqlalchemy import create_engine
   # import logging
   ```

2. **Type hints on EVERY function.** No exceptions.

3. **Pydantic models for ALL API inputs/outputs.** Never return raw dicts.

4. **RLS tenant isolation via aumos-common.** Never write raw SQL that bypasses RLS.

5. **Structured logging via structlog.** Never use print() or logging.getLogger().

6. **Publish domain events to Kafka after state changes.**

7. **Async by default.** All I/O operations must be async.

8. **Google-style docstrings** on all public classes and functions.

## Table Prefix

All ORM tables use the `fai_` prefix:
- `fai_assessments`   — FairnessAssessment (per-model bias assessment runs)
- `fai_bias_metrics`  — BiasMetric (individual metric results per assessment)
- `fai_mitigation_jobs` — MitigationJob (debiasing job runs with before/after metrics)
- `fai_monitors`      — FairnessMonitor (continuous monitoring configurations)

## Domain Responsibilities

- **Bias detection**: disparate impact, equal opportunity, demographic parity, equalized
  odds, calibration metrics via AIF360 and Fairlearn
- **Pre-processing mitigation**: reweighting (instance reweighting to equalize label rates
  across groups), uniform sampling / rejection sampling
- **In-processing mitigation**: adversarial debiasing (learns representation invariant to
  protected attributes during model training)
- **Post-processing mitigation**: threshold optimization (Hardt et al.) to equalize TPR/FPR
  across protected attribute groups
- **Continuous monitoring**: cron-driven periodic fairness checks, alerting when metrics
  drift beyond configured thresholds
- **Synthetic bias amplification detection**: statistical comparison of demographic
  distributions and label rates between real training data and generated synthetic data
  to detect when a generative model amplifies existing biases
- **Regulatory reports**: generate structured fairness reports suitable for ECOA (Equal
  Credit Opportunity Act) and EU AI Act Article 9/10 compliance submissions

## Metrics Reference

### AIF360 Metrics (aif360_detector.py)
- **Disparate Impact**: ratio of positive outcome rates between unprivileged and privileged
  groups. Threshold: >= 0.8 (the "4/5 rule")
- **Statistical Parity Difference**: difference in positive outcome rates. Threshold: in [-0.1, 0.1]
- **Equal Opportunity Difference**: difference in true positive rates. Threshold: in [-0.1, 0.1]
- **Average Odds Difference**: average of TPR and FPR differences. Threshold: in [-0.1, 0.1]
- **Theil Index**: entropy-based individual fairness measure. Threshold: < 0.1

### Fairlearn Metrics (fairlearn_detector.py)
- **Demographic Parity Difference**: max difference in selection rates. Threshold: <= 0.1
- **Demographic Parity Ratio**: min ratio of selection rates. Threshold: >= 0.8
- **Equalized Odds Difference**: max of TPR and FPR differences. Threshold: <= 0.1
- **Equalized Odds Ratio**: min of TPR and FPR ratios. Threshold: >= 0.8

### Synthetic Bias Amplification (synthetic_bias_detector.py)
- **Distribution Shift**: KL divergence between real and synthetic protected attribute
  distributions. Threshold: < 0.05
- **Label Rate Disparity**: difference in positive label rates per group between real and
  synthetic. Threshold: < 0.05
- **Amplification Factor**: ratio of bias metric in synthetic vs. real. Threshold: < 1.2
  (synthetic should not amplify bias by more than 20%)

## Kafka Topics Published

- `fairness.assessment_complete` — assessment finished (pass or fail)
- `fairness.assessment_failed` — assessment result: bias detected above threshold
- `fairness.mitigation_complete` — mitigation job finished
- `fairness.monitor_alert` — continuous monitor detected threshold breach
- `fairness.synthetic_bias_detected` — synthetic data amplifies real-data bias

## Settings Prefix

All environment variables use the `AUMOS_FAIRNESS_` prefix (see `.env.example`).

## API Conventions

- All endpoints under `/api/v1/` prefix
- Auth: Bearer JWT token (validated by aumos-common)
- Tenant: `X-Tenant-ID` header (set by auth middleware)
- Request ID: `X-Request-ID` header (auto-generated if missing)
- Pagination: `?page=1&page_size=20&sort_by=created_at&sort_order=desc`
- Errors: Standard `ErrorResponse` from aumos-common
- Content-Type: `application/json` (always)

## Testing Requirements

- Minimum coverage: **80%** for core modules, **60%** for adapters
- Use mock ML models (sklearn LogisticRegression or DecisionTree) in unit tests
- Use synthetic demographic datasets in unit tests (no real PII)
- Import fixtures from `aumos_common.testing`
- Use `testcontainers` for integration tests with real PostgreSQL/Kafka

## What Claude Code Should NOT Do

1. **Do NOT reimplement anything in aumos-common.**
2. **Do NOT use print().** Use `get_logger(__name__)`.
3. **Do NOT return raw dicts from API endpoints.** Use Pydantic models.
4. **Do NOT write raw SQL.** Use SQLAlchemy ORM with BaseRepository.
5. **Do NOT hardcode configuration.** Use Pydantic Settings with env vars.
6. **Do NOT skip type hints.** Every function signature must be typed.
7. **Do NOT import AGPL/GPL packages.** AIF360 uses Apache-2.0; verify before adding deps.
8. **Do NOT put business logic in API routes.** Routes call services; services contain logic.
9. **Do NOT hardcode protected attribute names.** They are tenant-configurable at runtime.
10. **Do NOT bypass RLS.** If cross-tenant data is needed, use `get_db_session_no_tenant` and document why.
11. **Do NOT call AIF360 or Fairlearn synchronously in request handlers.** Wrap in asyncio.to_thread().
12. **Do NOT store actual dataset rows in the database.** Store only metadata, metrics, and aggregate statistics.
