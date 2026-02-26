# Contributing to aumos-fairness-suite

Thank you for your interest in contributing to the AumOS Fairness Suite.

## License Restrictions

**Important:** Contributions incorporating AGPL, GPL, or LGPL licensed code are
not accepted. All dependencies must be permissively licensed (Apache-2.0, MIT, BSD).
AI Fairness 360 is Apache-2.0 licensed. Fairlearn is MIT licensed. Both are approved.

## Getting Started

1. Fork the repository and create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Run the test suite before making changes:
   ```bash
   make test
   ```

## Development Workflow

### Code Style

- Python 3.11+ with strict type hints on all function signatures and return types
- Line length: 120 characters (enforced by ruff)
- Use `ruff` for linting and formatting
- Use `mypy` in strict mode for type checking
- Google-style docstrings on all public classes and functions

```bash
make format    # auto-format with ruff
make lint      # lint check
make typecheck # mypy strict
```

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add equalized odds ratio metric to fairlearn detector
fix: correct disparate impact calculation when privileged group is empty
refactor: extract threshold optimization into dedicated post-processor
docs: document ECOA report field mapping
test: add integration tests for reweighting pre-processor
chore: upgrade aif360 to 0.6.1
```

Commit messages must explain **why**, not just what.

### Adding a New Metric

1. Add the metric computation to the appropriate detector in `adapters/detection/`
2. Register the metric name as a `MetricName` enum variant in `api/schemas.py`
3. Add the default threshold in `settings.py`
4. Write a unit test with a known-biased synthetic dataset verifying the metric value
5. Update the metric reference table in `README.md`

### Adding a New Mitigation Algorithm

1. Implement the algorithm in the appropriate stage module under `adapters/mitigation/`
2. Register the algorithm name as an `AlgorithmName` enum variant in `api/schemas.py`
3. Wire the algorithm into `MitigationService.apply_mitigation()` in `core/services.py`
4. Write a unit test verifying that applying the algorithm reduces at least one bias metric
5. Update the Mitigation Strategies section in `README.md`

### Testing

- Write tests alongside new code, not as an afterthought
- Unit tests must not require a live database or Kafka connection
- Use synthetic demographic datasets generated in `tests/conftest.py`
- Never use real PII or real demographic data in tests
- Minimum coverage targets: 80% for `core/`, 60% for `adapters/`

### Data Handling Principles

- Never store actual dataset rows in the database — only metadata and aggregate statistics
- Protected attribute names are tenant-configurable at runtime; never hardcode them
- All test datasets must be artificially generated — no real demographic data

## Pull Request Process

1. Ensure `make all` (lint + typecheck + test) passes before opening a PR
2. PR description must include:
   - What bias problem the change addresses
   - Which metrics are affected
   - Test coverage for the change
3. At least one reviewer from the AumOS fairness team must approve
4. Squash-merge to `main` with a conventional commit message

## Security

Do not open public GitHub issues for security vulnerabilities. See [SECURITY.md](SECURITY.md).
