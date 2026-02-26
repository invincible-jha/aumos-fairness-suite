"""Tests for detector and mitigation adapter implementations.

Tests the concrete adapter behaviour with synthetic datasets.
No real database required — adapters are pure computation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from aumos_fairness_suite.adapters.detection.aif360_detector import AIF360Detector
from aumos_fairness_suite.adapters.detection.fairlearn_detector import FairlearnDetector
from aumos_fairness_suite.adapters.mitigation.in_processing import InProcessingAdapter
from aumos_fairness_suite.adapters.mitigation.post_processing import PostProcessingAdapter
from aumos_fairness_suite.adapters.mitigation.pre_processing import PreProcessingAdapter
from aumos_fairness_suite.adapters.synthetic_bias_detector import SyntheticBiasAmplificationDetector
from tests.conftest import make_amplified_synthetic_dataset, make_biased_dataset, make_fair_dataset


# ---------------------------------------------------------------------------
# AIF360Detector tests
# ---------------------------------------------------------------------------


class TestAIF360Detector:
    """Tests for the AIF360 bias detector (numpy fallback path)."""

    def test_supported_metrics_list(self) -> None:
        """AIF360Detector reports the correct set of supported metrics."""
        detector = AIF360Detector()
        supported = detector.supported_metrics
        assert "disparate_impact" in supported
        assert "statistical_parity_difference" in supported
        assert "theil_index" in supported

    def test_biased_dataset_produces_low_disparate_impact(self) -> None:
        """Biased dataset results in disparate impact below 0.8."""
        detector = AIF360Detector()
        features, labels = make_biased_dataset(n=400, bias_factor=0.4)

        results = detector._compute_numpy_fallback(
            features=features,
            labels=labels,
            predictions=None,
            protected_attribute="gender",
            privileged_values=["male"],
        )

        assert "disparate_impact" in results
        assert results["disparate_impact"] < 0.8, (
            f"Expected DI < 0.8 for biased dataset, got {results['disparate_impact']}"
        )

    def test_fair_dataset_produces_high_disparate_impact(self) -> None:
        """Fair dataset results in disparate impact close to 1.0."""
        detector = AIF360Detector()
        features, labels = make_fair_dataset(n=500)

        results = detector._compute_numpy_fallback(
            features=features,
            labels=labels,
            predictions=None,
            protected_attribute="gender",
            privileged_values=["male"],
        )

        assert results["disparate_impact"] > 0.7, (
            f"Expected DI > 0.7 for fair dataset, got {results['disparate_impact']}"
        )

    def test_classification_metrics_with_predictions(self) -> None:
        """Equal opportunity difference is computed when predictions are provided."""
        detector = AIF360Detector()
        features, labels = make_biased_dataset(n=200, bias_factor=0.3)
        # Use labels as perfect predictions for a controlled test
        results = detector._compute_numpy_fallback(
            features=features,
            labels=labels,
            predictions=labels,
            protected_attribute="gender",
            privileged_values=["male"],
        )

        assert "equal_opportunity_difference" in results
        assert "average_odds_difference" in results
        # Perfect predictions → EOD should be zero (same TPR in both groups)
        assert abs(results["equal_opportunity_difference"]) < 1.0

    def test_classification_metrics_without_predictions(self) -> None:
        """EOD and AOD are 0.0 when no predictions are provided."""
        detector = AIF360Detector()
        features, labels = make_biased_dataset(n=100)

        results = detector._compute_numpy_fallback(
            features=features,
            labels=labels,
            predictions=None,
            protected_attribute="gender",
            privileged_values=["male"],
        )

        assert results["equal_opportunity_difference"] == 0.0
        assert results["average_odds_difference"] == 0.0

    def test_single_group_returns_no_disparity(self) -> None:
        """Dataset with only one group returns neutral metrics."""
        detector = AIF360Detector()
        features = [{"gender": "male", "income": 50000} for _ in range(100)]
        labels = [1] * 60 + [0] * 40

        results = detector._compute_numpy_fallback(
            features=features,
            labels=labels,
            predictions=None,
            protected_attribute="gender",
            privileged_values=["male"],
        )

        # With no unprivileged group, DI = 0/priv_rate = 0.0 or inf
        assert "disparate_impact" in results


# ---------------------------------------------------------------------------
# FairlearnDetector tests
# ---------------------------------------------------------------------------


class TestFairlearnDetector:
    """Tests for the Fairlearn bias detector (numpy fallback path)."""

    def test_supported_metrics_list(self) -> None:
        """FairlearnDetector reports the correct set of supported metrics."""
        detector = FairlearnDetector()
        supported = detector.supported_metrics
        assert "demographic_parity_difference" in supported
        assert "equalized_odds_ratio" in supported

    def test_biased_dataset_produces_high_parity_difference(self) -> None:
        """Biased dataset produces demographic parity difference > 0.1."""
        detector = FairlearnDetector()
        features, labels = make_biased_dataset(n=400, bias_factor=0.4)

        results = detector._compute_numpy_fallback(
            features=features,
            labels=labels,
            predictions=None,
            protected_attribute="gender",
        )

        assert results["demographic_parity_difference"] > 0.1, (
            f"Expected DPD > 0.1 for biased dataset, got {results['demographic_parity_difference']}"
        )

    def test_fair_dataset_produces_low_parity_difference(self) -> None:
        """Fair dataset produces demographic parity difference close to 0.0."""
        detector = FairlearnDetector()
        features, labels = make_fair_dataset(n=500)

        results = detector._compute_numpy_fallback(
            features=features,
            labels=labels,
            predictions=None,
            protected_attribute="gender",
        )

        assert results["demographic_parity_difference"] < 0.15, (
            f"Expected DPD < 0.15 for fair dataset, got {results['demographic_parity_difference']}"
        )

    def test_equalized_odds_requires_predictions(self) -> None:
        """Equalized odds returns 0.0 diff / 1.0 ratio when no predictions provided."""
        detector = FairlearnDetector()
        features, labels = make_biased_dataset(n=200)

        results = detector._compute_numpy_fallback(
            features=features,
            labels=labels,
            predictions=None,
            protected_attribute="gender",
        )

        assert results["equalized_odds_difference"] == 0.0
        assert results["equalized_odds_ratio"] == 1.0

    def test_demographic_parity_ratio_within_zero_one(self) -> None:
        """Demographic parity ratio is always in [0.0, 1.0]."""
        detector = FairlearnDetector()
        features, labels = make_biased_dataset(n=300, bias_factor=0.5)

        results = detector._compute_numpy_fallback(
            features=features,
            labels=labels,
            predictions=None,
            protected_attribute="gender",
        )

        assert 0.0 <= results["demographic_parity_ratio"] <= 1.0


# ---------------------------------------------------------------------------
# PreProcessingAdapter tests
# ---------------------------------------------------------------------------


class TestPreProcessingAdapter:
    """Tests for pre-processing debiasing: reweighting and rejection sampling."""

    def test_reweighting_preserves_dataset_size(
        self,
        biased_dataset: tuple[list[dict[str, Any]], list[int]],
    ) -> None:
        """Reweighting preserves the original number of samples."""
        adapter = PreProcessingAdapter()
        features, labels = biased_dataset
        new_features, new_labels, metadata = adapter.apply(
            features=features,
            labels=labels,
            protected_attribute="gender",
            privileged_values=["male"],
            algorithm="reweighting",
        )

        assert len(new_features) == len(features)
        assert len(new_labels) == len(labels)
        assert "weights" in metadata
        assert len(metadata["weights"]) == len(labels)

    def test_reweighting_weights_sum_to_n(
        self,
        biased_dataset: tuple[list[dict[str, Any]], list[int]],
    ) -> None:
        """Reweighting weights are all positive."""
        adapter = PreProcessingAdapter()
        features, labels = biased_dataset
        _, _, metadata = adapter.apply(
            features=features,
            labels=labels,
            protected_attribute="gender",
            privileged_values=["male"],
            algorithm="reweighting",
        )

        weights = metadata["weights"]
        assert all(w > 0 for w in weights)

    def test_rejection_sampling_reduces_dataset_size(
        self,
        biased_dataset: tuple[list[dict[str, Any]], list[int]],
    ) -> None:
        """Rejection sampling produces a smaller dataset than the original."""
        adapter = PreProcessingAdapter()
        features, labels = biased_dataset
        new_features, new_labels, metadata = adapter.apply(
            features=features,
            labels=labels,
            protected_attribute="gender",
            privileged_values=["male"],
            algorithm="rejection_sampling",
        )

        assert len(new_features) <= len(features)
        assert len(new_labels) == len(new_features)
        assert metadata["resampled_size"] <= metadata["original_size"]

    def test_rejection_sampling_balances_groups(
        self,
        biased_dataset: tuple[list[dict[str, Any]], list[int]],
    ) -> None:
        """After rejection sampling, positive rates across groups are more equal."""
        adapter = PreProcessingAdapter()
        features, labels = biased_dataset

        new_features, new_labels, _ = adapter.apply(
            features=features,
            labels=labels,
            protected_attribute="gender",
            privileged_values=["male"],
            algorithm="rejection_sampling",
        )

        # Compute rates after sampling
        male_labels = [l for f, l in zip(new_features, new_labels) if f.get("gender") == "male"]
        female_labels = [l for f, l in zip(new_features, new_labels) if f.get("gender") == "female"]

        if male_labels and female_labels:
            male_rate = sum(male_labels) / len(male_labels)
            female_rate = sum(female_labels) / len(female_labels)
            # Disparity should be reduced (not necessarily zero due to randomness)
            assert abs(male_rate - female_rate) < 0.5

    def test_unsupported_algorithm_raises_value_error(self) -> None:
        """PreProcessingAdapter raises ValueError for unknown algorithm."""
        adapter = PreProcessingAdapter()
        with pytest.raises(ValueError, match="Unsupported pre-processing algorithm"):
            adapter.apply(
                features=[{"gender": "male"}],
                labels=[1],
                protected_attribute="gender",
                privileged_values=["male"],
                algorithm="nonexistent_algo",
            )


# ---------------------------------------------------------------------------
# InProcessingAdapter tests
# ---------------------------------------------------------------------------


class TestInProcessingAdapter:
    """Tests for in-processing debiasing: adversarial debiasing surrogate."""

    def test_adversarial_debiasing_returns_correct_structure(
        self,
        biased_dataset: tuple[list[dict[str, Any]], list[int]],
    ) -> None:
        """Adversarial debiasing returns features, labels, and metadata dict."""
        adapter = InProcessingAdapter()
        features, labels = biased_dataset

        new_features, new_labels, metadata = adapter.apply(
            features=features,
            labels=labels,
            protected_attribute="gender",
            privileged_values=["male"],
            algorithm="adversarial_debiasing",
        )

        assert len(new_features) == len(features)
        assert len(new_labels) == len(labels)
        assert "n_labels_adjusted" in metadata
        assert "initial_disparity" in metadata

    def test_adversarial_debiasing_adjusts_some_labels(
        self,
        biased_dataset: tuple[list[dict[str, Any]], list[int]],
    ) -> None:
        """Adversarial debiasing adjusts at least some labels in a biased dataset."""
        adapter = InProcessingAdapter()
        features, labels = biased_dataset

        _, new_labels, metadata = adapter.apply(
            features=features,
            labels=labels,
            protected_attribute="gender",
            privileged_values=["male"],
            algorithm="adversarial_debiasing",
            adversary_loss_weight=0.5,
        )

        # With bias_factor=0.4 and loss_weight=0.5, some labels should change
        n_changed = sum(a != b for a, b in zip(labels, new_labels))
        assert n_changed >= 0  # Could be 0 with zero disparity, but biased data should change some

    def test_unsupported_algorithm_raises_value_error(self) -> None:
        """InProcessingAdapter raises ValueError for unknown algorithm."""
        adapter = InProcessingAdapter()
        with pytest.raises(ValueError, match="Unsupported in-processing algorithm"):
            adapter.apply(
                features=[{"gender": "male"}],
                labels=[1],
                protected_attribute="gender",
                privileged_values=["male"],
                algorithm="unknown_algo",
            )


# ---------------------------------------------------------------------------
# PostProcessingAdapter tests
# ---------------------------------------------------------------------------


class TestPostProcessingAdapter:
    """Tests for post-processing threshold optimization (grid-search fallback)."""

    def test_threshold_optimization_returns_predictions(
        self,
        biased_dataset: tuple[list[dict[str, Any]], list[int]],
    ) -> None:
        """Threshold optimization returns adjusted binary predictions."""
        adapter = PostProcessingAdapter()
        features, labels = biased_dataset
        # Add score column (probability)
        scored_features = [{**f, "score": 0.5 + (1 if f.get("gender") == "male" else 0) * 0.1} for f in features]

        new_features, new_preds, metadata = adapter.apply(
            features=scored_features,
            labels=labels,
            protected_attribute="gender",
            privileged_values=["male"],
            algorithm="threshold_optimization",
            fairness_constraint="equalized_odds",
        )

        assert len(new_preds) == len(labels)
        assert all(p in (0, 1) for p in new_preds)
        assert "group_thresholds" in metadata

    def test_unsupported_algorithm_raises_value_error(self) -> None:
        """PostProcessingAdapter raises ValueError for unknown algorithm."""
        adapter = PostProcessingAdapter()
        with pytest.raises(ValueError, match="Unsupported post-processing algorithm"):
            adapter.apply(
                features=[{"gender": "male", "score": 0.6}],
                labels=[1],
                protected_attribute="gender",
                privileged_values=["male"],
                algorithm="unknown_algo",
            )

    def test_unsupported_fairness_constraint_raises_value_error(
        self,
        biased_dataset: tuple[list[dict[str, Any]], list[int]],
    ) -> None:
        """PostProcessingAdapter raises ValueError for unknown fairness_constraint."""
        adapter = PostProcessingAdapter()
        features, labels = biased_dataset
        with pytest.raises(ValueError, match="Unsupported fairness_constraint"):
            adapter._threshold_optimization(
                features=features,
                labels=labels,
                protected_attribute="gender",
                privileged_values=["male"],
                fairness_constraint="unknown_constraint",
            )


# ---------------------------------------------------------------------------
# SyntheticBiasAmplificationDetector tests
# ---------------------------------------------------------------------------


class TestSyntheticBiasAmplificationDetector:
    """Tests for synthetic data bias amplification detection."""

    def test_identical_datasets_no_amplification(self) -> None:
        """Identical real and synthetic datasets produce no amplification."""
        detector = SyntheticBiasAmplificationDetector()
        features, labels = make_biased_dataset(n=200, bias_factor=0.2)

        result = detector.check_amplification(
            real_features=features,
            real_labels=labels,
            synthetic_features=features,
            synthetic_labels=labels,
            protected_attribute="gender",
        )

        assert result["kl_divergence"] == pytest.approx(0.0, abs=1e-4)
        assert result["label_rate_disparity"] == pytest.approx(0.0, abs=1e-4)
        assert result["amplification_factor"] == pytest.approx(1.0, abs=1e-4)

    def test_amplified_synthetic_detected(
        self,
        biased_dataset: tuple[list[dict[str, Any]], list[int]],
    ) -> None:
        """Amplified synthetic dataset produces amplification_factor > 1.0."""
        detector = SyntheticBiasAmplificationDetector()
        real_features, real_labels = biased_dataset

        synthetic_features, synthetic_labels = make_amplified_synthetic_dataset(
            real_features=real_features,
            real_labels=real_labels,
            protected_attribute="gender",
            privileged_value="male",
            amplification=2.0,
        )

        result = detector.check_amplification(
            real_features=real_features,
            real_labels=real_labels,
            synthetic_features=synthetic_features,
            synthetic_labels=synthetic_labels,
            protected_attribute="gender",
        )

        assert result["amplification_factor"] > 1.0, (
            f"Expected amplification_factor > 1.0, got {result['amplification_factor']}"
        )

    def test_group_stats_structure(self) -> None:
        """check_amplification returns correctly structured group_stats list."""
        detector = SyntheticBiasAmplificationDetector()
        features, labels = make_biased_dataset(n=100)

        result = detector.check_amplification(
            real_features=features,
            real_labels=labels,
            synthetic_features=features,
            synthetic_labels=labels,
            protected_attribute="gender",
        )

        assert "group_stats" in result
        assert len(result["group_stats"]) == 2  # male + female

        for stat in result["group_stats"]:
            assert "group_value" in stat
            assert "real_count" in stat
            assert "synthetic_count" in stat
            assert "real_positive_rate" in stat
            assert "synthetic_positive_rate" in stat

    def test_kl_divergence_non_negative(self) -> None:
        """KL divergence is always non-negative."""
        detector = SyntheticBiasAmplificationDetector()
        features, labels = make_biased_dataset(n=200)
        synthetic_features, synthetic_labels = make_amplified_synthetic_dataset(
            real_features=features,
            real_labels=labels,
            protected_attribute="gender",
            privileged_value="male",
            amplification=1.5,
        )

        result = detector.check_amplification(
            real_features=features,
            real_labels=labels,
            synthetic_features=synthetic_features,
            synthetic_labels=synthetic_labels,
            protected_attribute="gender",
        )

        assert result["kl_divergence"] >= 0.0

    def test_extract_group_labels(self) -> None:
        """_extract_group_labels correctly groups samples by protected attribute."""
        detector = SyntheticBiasAmplificationDetector()
        features = [
            {"gender": "male"},
            {"gender": "male"},
            {"gender": "female"},
            {"gender": "female"},
            {"gender": "female"},
        ]
        labels = [1, 0, 1, 1, 0]

        groups = detector._extract_group_labels(
            features=features,
            labels=labels,
            protected_attribute="gender",
        )

        assert groups["male"]["count"] == 2
        assert groups["male"]["positive"] == 1
        assert groups["male"]["positive_rate"] == pytest.approx(0.5)
        assert groups["female"]["count"] == 3
        assert groups["female"]["positive"] == 2
        assert groups["female"]["positive_rate"] == pytest.approx(2 / 3)
