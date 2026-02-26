# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-26

### Added

- Initial scaffolding for aumos-fairness-suite
- FastAPI service with hexagonal architecture (api/core/adapters)
- SQLAlchemy ORM models with `fai_` table prefix:
  - `FairnessAssessment` — per-model bias assessment runs
  - `BiasMetric` — individual metric results per assessment
  - `MitigationJob` — debiasing job runs with before/after metrics
  - `FairnessMonitor` — continuous monitoring configurations
- AI Fairness 360 detector (`aif360_detector.py`): disparate impact, statistical parity
  difference, equal opportunity difference, average odds difference, Theil index
- Fairlearn detector (`fairlearn_detector.py`): demographic parity difference/ratio,
  equalized odds difference/ratio
- Pre-processing mitigation (`pre_processing.py`): reweighting and rejection sampling
- In-processing mitigation (`in_processing.py`): adversarial debiasing skeleton
- Post-processing mitigation (`post_processing.py`): threshold optimization (Hardt et al.)
- Synthetic bias amplification detector (`synthetic_bias_detector.py`): KL divergence,
  label rate disparity, amplification factor computation
- Continuous fairness monitoring service with cron-based scheduling
- Regulatory report generation for ECOA and EU AI Act Article 9/10
- Kafka event publisher for all fairness lifecycle events
- Full REST API under `/api/v1/`:
  - `POST /assessments` — run fairness assessment
  - `GET /assessments` — list assessments (paginated)
  - `GET /assessments/{id}` — get assessment with metrics
  - `POST /mitigations` — apply mitigation strategy
  - `GET /mitigations/{id}` — get mitigation results
  - `POST /monitors` — create continuous fairness monitor
  - `GET /monitors` — list monitors
  - `POST /synthetic-bias-check` — check synthetic data for bias amplification
  - `GET /reports/{assessment_id}` — generate regulatory fairness report
- Docker multi-stage build with non-root user and health check
- docker-compose.dev.yml with postgres, kafka, and app service
- CI/CD pipeline: lint, typecheck, test, docker build, license check
- Standard deliverables: CLAUDE.md, README, pyproject.toml, Makefile, .env.example,
  CONTRIBUTING.md, SECURITY.md, .gitignore, .dockerignore, LICENSE
