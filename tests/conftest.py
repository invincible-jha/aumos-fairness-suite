"""Shared test fixtures for aumos-fairness-suite.

Uses synthetic demographic datasets generated with numpy — no real PII.
All infrastructure (database, Kafka) is mocked in unit tests.
Integration tests would use testcontainers for real PostgreSQL.
"""

from __future__ import annotations

import random
import uuid
from typing import Any, Generator

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Random seed for reproducible synthetic data
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Synthetic dataset factories
# ---------------------------------------------------------------------------


def make_biased_dataset(
    n: int = 200,
    bias_factor: float = 0.3,
    protected_attribute: str = "gender",
    privileged_value: str = "male",
    unprivileged_value: str = "female",
    label_column: str = "approved",
    seed: int = RANDOM_SEED,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Generate a biased binary classification dataset.

    The privileged group receives positive labels at rate (0.5 + bias_factor/2)
    while the unprivileged group receives positive labels at rate (0.5 - bias_factor/2).
    Disparate impact = unprivileged_rate / privileged_rate < 1.0.

    Args:
        n: Total number of samples.
        bias_factor: Difference in positive rates between groups (0 = fair, 1 = max bias).
        protected_attribute: Name of the protected attribute column.
        privileged_value: Value for the privileged group.
        unprivileged_value: Value for the unprivileged group.
        label_column: Name of the target label column (unused in return — labels separate).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (features_list, labels_list).
    """
    local_rng = np.random.default_rng(seed)
    half = n // 2
    features: list[dict[str, Any]] = []
    labels: list[int] = []

    # Privileged group
    priv_positive_rate = 0.5 + bias_factor / 2.0
    for _ in range(half):
        features.append({
            protected_attribute: privileged_value,
            "age": int(local_rng.integers(25, 65)),
            "income": float(local_rng.uniform(30000, 120000)),
            "credit_score": int(local_rng.integers(600, 850)),
        })
        labels.append(int(local_rng.random() < priv_positive_rate))

    # Unprivileged group
    unpriv_positive_rate = 0.5 - bias_factor / 2.0
    for _ in range(n - half):
        features.append({
            protected_attribute: unprivileged_value,
            "age": int(local_rng.integers(25, 65)),
            "income": float(local_rng.uniform(30000, 120000)),
            "credit_score": int(local_rng.integers(600, 850)),
        })
        labels.append(int(local_rng.random() < unpriv_positive_rate))

    return features, labels


def make_fair_dataset(
    n: int = 200,
    protected_attribute: str = "gender",
    privileged_value: str = "male",
    unprivileged_value: str = "female",
    seed: int = RANDOM_SEED,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Generate a fair dataset with equal positive rates across groups.

    Args:
        n: Total number of samples.
        protected_attribute: Name of the protected attribute column.
        privileged_value: Value for the privileged group.
        unprivileged_value: Value for the unprivileged group.
        seed: Random seed.

    Returns:
        Tuple of (features_list, labels_list).
    """
    return make_biased_dataset(
        n=n,
        bias_factor=0.0,
        protected_attribute=protected_attribute,
        privileged_value=privileged_value,
        unprivileged_value=unprivileged_value,
        seed=seed,
    )


def make_amplified_synthetic_dataset(
    real_features: list[dict[str, Any]],
    real_labels: list[int],
    protected_attribute: str,
    privileged_value: str,
    amplification: float = 1.5,
    seed: int = RANDOM_SEED,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Generate a synthetic dataset that amplifies the bias in real data.

    Increases the disparity between privileged and unprivileged positive rates
    by the amplification factor to simulate a biased generative model.

    Args:
        real_features: Real dataset feature dicts.
        real_labels: Real dataset binary labels.
        protected_attribute: Protected attribute column name.
        privileged_value: Privileged group value.
        amplification: How much to amplify the existing disparity (1.0 = no change).
        seed: Random seed.

    Returns:
        Tuple of (synthetic features, synthetic labels) with amplified bias.
    """
    local_rng = np.random.default_rng(seed)
    priv_mask = [f.get(protected_attribute) == privileged_value for f in real_features]
    labels_arr = np.array(real_labels, dtype=float)

    priv_rate = float(labels_arr[priv_mask].mean()) if any(priv_mask) else 0.5
    unpriv_rate = float(labels_arr[[not m for m in priv_mask]].mean()) if any(not m for m in priv_mask) else 0.5

    disparity = priv_rate - unpriv_rate
    amplified_priv_rate = min(1.0, priv_rate + disparity * (amplification - 1) / 2)
    amplified_unpriv_rate = max(0.0, unpriv_rate - disparity * (amplification - 1) / 2)

    synthetic_features = [dict(f) for f in real_features]
    synthetic_labels: list[int] = []

    for feat in synthetic_features:
        is_priv = feat.get(protected_attribute) == privileged_value
        rate = amplified_priv_rate if is_priv else amplified_unpriv_rate
        synthetic_labels.append(int(local_rng.random() < rate))

    return synthetic_features, synthetic_labels


# ---------------------------------------------------------------------------
# UUID helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tenant_id() -> uuid.UUID:
    """Return a fixed tenant UUID for use in tests."""
    return uuid.UUID("10000000-0000-0000-0000-000000000001")


@pytest.fixture
def model_id() -> uuid.UUID:
    """Return a fixed model UUID for use in tests."""
    return uuid.UUID("20000000-0000-0000-0000-000000000002")


@pytest.fixture
def dataset_id() -> uuid.UUID:
    """Return a fixed dataset UUID for use in tests."""
    return uuid.UUID("30000000-0000-0000-0000-000000000003")


@pytest.fixture
def assessment_id() -> uuid.UUID:
    """Return a fixed assessment UUID for use in tests."""
    return uuid.UUID("40000000-0000-0000-0000-000000000004")


# ---------------------------------------------------------------------------
# Dataset fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def biased_dataset() -> tuple[list[dict[str, Any]], list[int]]:
    """Biased dataset with gender disparity (bias_factor=0.4 → DI ~ 0.6)."""
    return make_biased_dataset(n=300, bias_factor=0.4)


@pytest.fixture
def fair_dataset() -> tuple[list[dict[str, Any]], list[int]]:
    """Fair dataset with equal positive rates across gender groups."""
    return make_fair_dataset(n=300)


@pytest.fixture
def biased_dataset_rows() -> list[dict[str, Any]]:
    """Biased dataset as a single list of row dicts with label column included."""
    features, labels = make_biased_dataset(n=200, bias_factor=0.4)
    return [{**feat, "approved": label} for feat, label in zip(features, labels)]


@pytest.fixture
def amplified_synthetic_rows(biased_dataset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Synthetic dataset rows with amplified bias relative to biased_dataset_rows."""
    real_features = [{k: v for k, v in row.items() if k != "approved"} for row in biased_dataset_rows]
    real_labels = [row["approved"] for row in biased_dataset_rows]
    synthetic_features, synthetic_labels = make_amplified_synthetic_dataset(
        real_features=real_features,
        real_labels=real_labels,
        protected_attribute="gender",
        privileged_value="male",
        amplification=1.8,
    )
    return [{**feat, "approved": label} for feat, label in zip(synthetic_features, synthetic_labels)]
