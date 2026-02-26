# aumos-fairness-suite

> Bias detection and mitigation with AIF360, Fairlearn, continuous monitoring, and synthetic data bias amplification detection for AumOS Enterprise.

[![CI](https://github.com/aumos-enterprise/aumos-fairness-suite/actions/workflows/ci.yml/badge.svg)](https://github.com/aumos-enterprise/aumos-fairness-suite/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Bias Metrics](#bias-metrics)
4. [Mitigation Strategies](#mitigation-strategies)
5. [Synthetic Bias Detection](#synthetic-bias-detection)
6. [API Reference](#api-reference)
7. [Configuration](#configuration)
8. [Local Development](#local-development)
9. [Testing](#testing)
10. [Regulatory Compliance](#regulatory-compliance)

---

## 1. Overview

`aumos-fairness-suite` is repo #27 in the AumOS Enterprise platform. It provides
end-to-end algorithmic fairness tooling across the entire ML lifecycle:

- **Assessment**: measure bias across protected attribute groups using IBM AI Fairness 360
  and Microsoft Fairlearn metric suites
- **Mitigation**: apply pre-, in-, or post-processing debiasing strategies to models and
  datasets without manual intervention
- **Monitoring**: continuous cron-driven fairness checks on deployed models with alerting
  when metrics drift above configured thresholds
- **Synthetic bias amplification detection**: statistical test suite that verifies
  AI-generated synthetic datasets do not amplify demographic biases present in real data
- **Regulatory reporting**: auto-generate structured reports for ECOA (Equal Credit
  Opportunity Act) and EU AI Act Article 9/10 compliance workflows

This service is consumed primarily by `aumos-governance-engine` as part of the
automated model approval pipeline. Failed fairness assessments block model promotion
and emit `fairness.assessment_failed` Kafka events that trigger retraining workflows
in `aumos-mlops-lifecycle`.

---

## 2. Architecture

The service follows a hexagonal architecture (ports and adapters):

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Layer (api/)                      │
│   router.py — thin HTTP handlers, auth via aumos-common          │
│   schemas.py — Pydantic request/response models                  │
└───────────────────┬─────────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────────┐
│                      Core Layer (core/)                          │
│   services.py — BiasDetectionService, MitigationService,         │
│                 MonitoringService, ReportingService               │
│   interfaces.py — Protocol definitions consumed by services      │
│   models.py — SQLAlchemy ORM models (fai_ table prefix)          │
└───────────────────┬─────────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────────┐
│                    Adapters Layer (adapters/)                    │
│   repositories.py — BaseRepository extensions                    │
│   kafka.py — FairnessEventPublisher                              │
│   detection/                                                     │
│     aif360_detector.py — IBM AI Fairness 360 metrics             │
│     fairlearn_detector.py — Microsoft Fairlearn metrics          │
│   mitigation/                                                    │
│     pre_processing.py — Reweighting, rejection sampling          │
│     in_processing.py — Adversarial debiasing                     │
│     post_processing.py — Threshold optimization (Hardt et al.)   │
│   synthetic_bias_detector.py — Amplification detection           │
└─────────────────────────────────────────────────────────────────┘
```

**Position in AumOS graph:**

```
aumos-common + aumos-proto
        ↓
aumos-fairness-suite
        ↓
aumos-governance-engine → fairness reports, assessment status
aumos-mlops-lifecycle   → retraining triggers on bias failures
aumos-observability     → fairness metric dashboards
```

---

## 3. Bias Metrics

### AI Fairness 360 Metrics

| Metric | Definition | Pass Threshold |
|--------|------------|----------------|
| Disparate Impact | P(Y=1 \| unprivileged) / P(Y=1 \| privileged) | >= 0.8 |
| Statistical Parity Difference | P(Y=1 \| unprivileged) - P(Y=1 \| privileged) | [-0.1, 0.1] |
| Equal Opportunity Difference | TPR difference across groups | [-0.1, 0.1] |
| Average Odds Difference | Avg of TPR and FPR differences | [-0.1, 0.1] |
| Theil Index | Entropy-based individual fairness | < 0.1 |

### Fairlearn Metrics

| Metric | Definition | Pass Threshold |
|--------|------------|----------------|
| Demographic Parity Difference | Max selection rate difference | <= 0.1 |
| Demographic Parity Ratio | Min selection rate ratio | >= 0.8 |
| Equalized Odds Difference | Max of TPR and FPR differences | <= 0.1 |
| Equalized Odds Ratio | Min of TPR and FPR ratios | >= 0.8 |

---

## 4. Mitigation Strategies

### Pre-Processing

Applied to the training dataset before model training:

- **Reweighting** (`algorithm: reweighting`): assigns instance weights to equalize
  label rates across protected attribute groups, preserving full dataset size
- **Rejection Sampling** (`algorithm: rejection_sampling`): resamples the dataset
  to balance representation while maintaining marginal distributions

### In-Processing

Applied during model training:

- **Adversarial Debiasing** (`algorithm: adversarial_debiasing`): trains a predictor
  and adversary jointly; the adversary tries to predict the protected attribute from
  the predictor's output, forcing the predictor to learn bias-invariant representations

### Post-Processing

Applied to model predictions at inference time:

- **Threshold Optimization** (`algorithm: threshold_optimization`): learns per-group
  decision thresholds that equalize a chosen fairness criterion (equalized odds,
  demographic parity) while maximizing overall accuracy

---

## 5. Synthetic Bias Detection

When synthetic data generators (CTGAN, TVAE, GaussianCopula) are used to augment
training data or replace real data, they can inadvertently amplify demographic biases
present in the original dataset.

`aumos-fairness-suite` provides a dedicated `POST /api/v1/synthetic-bias-check`
endpoint that:

1. Computes protected attribute distributions in both real and synthetic datasets
2. Measures KL divergence of group-conditional label distributions
3. Computes per-group positive label rates in real vs. synthetic
4. Calculates the **amplification factor** (bias metric in synthetic / bias metric in real)
5. Returns a structured report with pass/fail per check and overall verdict

An amplification factor > 1.2 (synthetic amplifies real bias by more than 20%) triggers
a `fairness.synthetic_bias_detected` Kafka event and blocks downstream usage of the
synthetic dataset via `aumos-governance-engine`.

---

## 6. API Reference

All endpoints require Bearer JWT authentication and `X-Tenant-ID` header.

### Assessments

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/assessments` | Run a fairness assessment on a model + dataset |
| GET | `/api/v1/assessments` | List assessments (paginated) |
| GET | `/api/v1/assessments/{id}` | Get assessment with all bias metrics |

### Mitigations

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/mitigations` | Apply a mitigation strategy |
| GET | `/api/v1/mitigations/{id}` | Get mitigation job results |

### Monitors

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/monitors` | Create a continuous fairness monitor |
| GET | `/api/v1/monitors` | List monitors |

### Synthetic Bias

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/synthetic-bias-check` | Check synthetic data for bias amplification |

### Reports

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/reports/{assessment_id}` | Generate regulatory fairness report |

---

## 7. Configuration

All configuration uses environment variables with the `AUMOS_FAIRNESS_` prefix.
Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `AUMOS_FAIRNESS_DATABASE__URL` | — | PostgreSQL async connection URL |
| `AUMOS_FAIRNESS_KAFKA__BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka broker(s) |
| `AUMOS_FAIRNESS_DISPARATE_IMPACT_THRESHOLD` | `0.8` | Pass threshold for disparate impact |
| `AUMOS_FAIRNESS_PARITY_DIFFERENCE_THRESHOLD` | `0.1` | Max allowed parity difference |
| `AUMOS_FAIRNESS_AMPLIFICATION_THRESHOLD` | `1.2` | Max allowed synthetic amplification |
| `AUMOS_FAIRNESS_GOVERNANCE_ENGINE_URL` | `http://localhost:8016` | Governance engine base URL |
| `AUMOS_FAIRNESS_MLOPS_URL` | `http://localhost:8008` | MLOps lifecycle base URL |
| `AUMOS_LOG_LEVEL` | `info` | Structured log level |

---

## 8. Local Development

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- PostgreSQL 16 (via Docker)
- Kafka (via Docker)

### Quick Start

```bash
# Clone and install
git clone https://github.com/aumos-enterprise/aumos-fairness-suite
cd aumos-fairness-suite
pip install -e ".[dev]"

# Start infrastructure
docker compose -f docker-compose.dev.yml up -d

# Copy and configure environment
cp .env.example .env

# Run the service
uvicorn aumos_fairness_suite.main:app --reload --port 8017
```

### Makefile Targets

```bash
make install     # Install with dev extras
make test        # Run test suite with coverage
make test-quick  # Fast test run (no coverage)
make lint        # ruff check + format check
make format      # ruff auto-format
make typecheck   # mypy strict mode
make docker-build # Build Docker image
make docker-run  # Start via docker-compose
make clean       # Remove build artifacts
```

---

## 9. Testing

```bash
# Full suite with coverage
pytest tests/ -v --cov=aumos_fairness_suite --cov-report=term-missing

# Unit tests only (no infrastructure required)
pytest tests/test_services.py tests/test_api.py -v

# Integration tests (requires Docker)
pytest tests/test_repositories.py -v
```

Coverage targets: **80%** for `core/`, **60%** for `adapters/`.

Tests use synthetic demographic datasets generated in `conftest.py` — no real PII
is used anywhere in the test suite.

---

## 10. Regulatory Compliance

### ECOA (Equal Credit Opportunity Act)

The `GET /api/v1/reports/{assessment_id}` endpoint generates a structured JSON report
that satisfies ECOA adverse action analysis requirements:

- Disparate impact ratios per protected class (race, gender, age, national origin)
- Statistical significance tests for observed disparities
- Mitigation actions applied and resulting metric changes
- Assessment timestamp and model version for audit trail

### EU AI Act (Article 9 / Article 10)

For high-risk AI systems under the EU AI Act:

- **Article 9** (Risk management): assessments document identified bias risks and
  mitigation measures applied
- **Article 10** (Data governance): synthetic bias checks verify training data
  quality with respect to demographic representation

All assessment results are stored with full audit trails (tenant, model version,
dataset ID, timestamp) suitable for regulatory submission. Reports are versioned
and immutable once generated.

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

> AumOS Enterprise — Composable AI Platform
