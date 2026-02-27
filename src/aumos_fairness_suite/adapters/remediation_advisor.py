"""Remediation Advisor — actionable debiasing strategy recommendations with impact estimates.

Analyzes bias assessment results and intersectional disparity data to produce a
prioritized, actionable debiasing roadmap. Recommends specific mitigation strategies
with estimated impact, implementation difficulty, and projected before/after metric values.

Remediation strategies:
- Pre-processing: resampling (oversample underrepresented groups), reweighting
  (adjust instance weights), synthetic augmentation
- In-processing: adversarial debiasing, fairness constraints, regularization terms
- Post-processing: threshold adjustment per group, calibration, equalized odds optimization
- Data collection: identify gaps in representation and recommend targeted collection
- Feature engineering: detect and remove proxy features, suggest fair representations

Outputs are suitable for regulatory response documentation (ECOA, EU AI Act Article 9/10).
"""

from dataclasses import dataclass, field
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)


# Remediation strategy catalog with properties
REMEDIATION_STRATEGIES: dict[str, dict[str, Any]] = {
    "targeted_resampling": {
        "category": "pre_processing",
        "name": "Targeted Resampling",
        "description": (
            "Oversample underrepresented intersectional groups using SMOTE or "
            "similar techniques. Increases minority group representation in training data."
        ),
        "implementation_difficulty": "low",
        "expected_parity_reduction": 0.3,  # Fraction of gap expected to be reduced
        "expected_odds_reduction": 0.2,
        "side_effects": [
            "May reduce overall model accuracy slightly",
            "Can introduce synthetic noise if oversample ratio > 5x",
        ],
        "libraries": ["imbalanced-learn"],
        "time_estimate_days": 2,
    },
    "instance_reweighting": {
        "category": "pre_processing",
        "name": "Instance Reweighting",
        "description": (
            "Assign higher training weights to underrepresented group samples. "
            "Adjusts group label rates without modifying the dataset."
        ),
        "implementation_difficulty": "low",
        "expected_parity_reduction": 0.25,
        "expected_odds_reduction": 0.25,
        "side_effects": [
            "May cause overfitting on weighted groups in small datasets",
        ],
        "libraries": ["aif360", "custom"],
        "time_estimate_days": 1,
    },
    "proxy_feature_removal": {
        "category": "feature_engineering",
        "name": "Proxy Feature Removal",
        "description": (
            "Remove features that are highly correlated with protected attributes. "
            "Eliminates proxy discrimination pathways from the model."
        ),
        "implementation_difficulty": "medium",
        "expected_parity_reduction": 0.4,
        "expected_odds_reduction": 0.35,
        "side_effects": [
            "May reduce predictive accuracy if removed features carry legitimate signal",
            "Requires careful correlation analysis to avoid removing too many features",
        ],
        "libraries": ["numpy", "pandas"],
        "time_estimate_days": 3,
    },
    "adversarial_debiasing": {
        "category": "in_processing",
        "name": "Adversarial Debiasing",
        "description": (
            "Train a model that simultaneously maximises task accuracy and minimises "
            "an adversary's ability to predict the protected attribute from the model's "
            "learned representation. Learns a fair representation during training."
        ),
        "implementation_difficulty": "high",
        "expected_parity_reduction": 0.5,
        "expected_odds_reduction": 0.45,
        "side_effects": [
            "Requires model retraining from scratch",
            "Adds adversarial training complexity (2x training time typical)",
            "Hyperparameter-sensitive — requires careful tuning of adversary strength",
        ],
        "libraries": ["aif360", "torch"],
        "time_estimate_days": 14,
    },
    "fairness_constraints": {
        "category": "in_processing",
        "name": "Fairness Constraint Optimization",
        "description": (
            "Add fairness constraints (demographic parity, equalized odds) as regularization "
            "terms during model training via reduction-based approaches (e.g., Exponentiated "
            "Gradient, Grid Search over Lagrange multipliers)."
        ),
        "implementation_difficulty": "high",
        "expected_parity_reduction": 0.45,
        "expected_odds_reduction": 0.5,
        "side_effects": [
            "Requires model retraining from scratch",
            "Trade-off between fairness and accuracy is explicit but requires tuning",
        ],
        "libraries": ["fairlearn"],
        "time_estimate_days": 10,
    },
    "threshold_adjustment": {
        "category": "post_processing",
        "name": "Per-Group Threshold Adjustment",
        "description": (
            "Adjust decision thresholds independently per protected group to equalize "
            "positive prediction rates or TPR/FPR ratios. Does not require model retraining."
        ),
        "implementation_difficulty": "low",
        "expected_parity_reduction": 0.6,
        "expected_odds_reduction": 0.55,
        "side_effects": [
            "Legally sensitive in some jurisdictions (explicit group-based decision rules)",
            "May reduce precision for groups receiving lower thresholds",
            "Requires careful monitoring for threshold drift",
        ],
        "libraries": ["fairlearn", "custom"],
        "time_estimate_days": 2,
    },
    "calibrated_thresholds": {
        "category": "post_processing",
        "name": "Calibrated Equalized Odds",
        "description": (
            "Post-process predicted probabilities using the Hardt et al. (2016) method "
            "to equalize TPR and FPR across groups while maximizing accuracy subject to "
            "the equalized odds constraint."
        ),
        "implementation_difficulty": "medium",
        "expected_parity_reduction": 0.5,
        "expected_odds_reduction": 0.7,
        "side_effects": [
            "May introduce randomization in predictions near decision boundary",
            "Optimal only for binary classification",
        ],
        "libraries": ["aif360"],
        "time_estimate_days": 3,
    },
    "targeted_data_collection": {
        "category": "data_collection",
        "name": "Targeted Data Collection",
        "description": (
            "Design a data collection program specifically targeting underrepresented "
            "intersectional groups. Improves model generalization across all groups."
        ),
        "implementation_difficulty": "high",
        "expected_parity_reduction": 0.35,
        "expected_odds_reduction": 0.3,
        "side_effects": [
            "Requires significant time and resources (weeks to months)",
            "May not be feasible for historical datasets",
        ],
        "libraries": [],
        "time_estimate_days": 60,
    },
    "synthetic_augmentation": {
        "category": "pre_processing",
        "name": "Fairness-Aware Synthetic Augmentation",
        "description": (
            "Generate synthetic training samples for underrepresented groups using "
            "a fairness-constrained generative model. Augments existing data without "
            "requiring new real-world data collection."
        ),
        "implementation_difficulty": "high",
        "expected_parity_reduction": 0.3,
        "expected_odds_reduction": 0.25,
        "side_effects": [
            "Risk of synthetic bias amplification — validate with aumos-fairness-suite",
            "Generated samples may not reflect true population distributions",
        ],
        "libraries": ["aumos-healthcare-synth", "custom"],
        "time_estimate_days": 7,
    },
}

# Strategy applicability rules: (condition) -> [recommended strategy ids]
STRATEGY_APPLICABILITY: list[tuple[str, list[str]]] = [
    # High parity gap → threshold adjustment is fastest fix
    ("high_parity_gap", ["threshold_adjustment", "calibrated_thresholds", "instance_reweighting"]),
    # High odds gap → focus on equalized odds approaches
    ("high_odds_gap", ["calibrated_thresholds", "fairness_constraints", "threshold_adjustment"]),
    # Proxy features detected → remove them
    ("proxy_features", ["proxy_feature_removal", "adversarial_debiasing"]),
    # Small group size → data collection or augmentation
    ("small_groups", ["targeted_data_collection", "synthetic_augmentation", "targeted_resampling"]),
    # Intersectional amplification detected → complex mitigation needed
    ("intersectional_amplification", ["fairness_constraints", "adversarial_debiasing", "targeted_resampling"]),
    # Multiple disparate groups → systematic approach
    ("many_disparate_groups", ["adversarial_debiasing", "fairness_constraints", "instance_reweighting"]),
]


@dataclass
class StrategyRecommendation:
    """A single recommended remediation strategy with impact estimates.

    Attributes:
        strategy_id: Identifier matching REMEDIATION_STRATEGIES keys.
        priority: Priority rank (1 = highest).
        rationale: Why this strategy was recommended for this bias pattern.
        estimated_parity_reduction: Expected reduction in parity gap (fraction).
        estimated_odds_reduction: Expected reduction in odds gap (fraction).
        projected_parity_gap_after: Projected parity gap after applying strategy.
        projected_odds_gap_after: Projected odds gap after applying strategy.
        implementation_difficulty: 'low', 'medium', or 'high'.
        time_estimate_days: Estimated implementation calendar days.
        required_libraries: Python packages needed.
        side_effects: Known trade-offs or risks.
        implementation_steps: Step-by-step implementation guidance.
    """

    strategy_id: str
    priority: int
    rationale: str
    estimated_parity_reduction: float
    estimated_odds_reduction: float
    projected_parity_gap_after: float
    projected_odds_gap_after: float
    implementation_difficulty: str
    time_estimate_days: int
    required_libraries: list[str]
    side_effects: list[str]
    implementation_steps: list[str] = field(default_factory=list)


class RemediationAdvisor:
    """Debiasing strategy recommender with impact estimation.

    Analyzes bias detection results and intersectional disparity findings to
    produce an actionable, prioritized debiasing roadmap. Strategies are ranked
    by expected impact, implementation feasibility, and time-to-deploy. Projected
    before/after metric values help compliance officers assess regulatory risk.

    Args:
        max_recommendations: Maximum strategies to include in the roadmap.
        prefer_quick_wins: Prefer low-difficulty strategies when True.
        require_no_retraining: Restrict to post-processing strategies if True.
        regulatory_context: Regulatory framework context ('ecoa', 'eu_ai_act', 'general').
    """

    def __init__(
        self,
        max_recommendations: int = 5,
        prefer_quick_wins: bool = False,
        require_no_retraining: bool = False,
        regulatory_context: str = "general",
    ) -> None:
        """Initialize the RemediationAdvisor.

        Args:
            max_recommendations: Maximum strategies to return.
            prefer_quick_wins: If True, deprioritize strategies requiring retraining.
            require_no_retraining: If True, only include post-processing strategies.
            regulatory_context: Regulatory framework to reference in guidance.
        """
        self._max_recommendations = max_recommendations
        self._prefer_quick_wins = prefer_quick_wins
        self._require_no_retraining = require_no_retraining
        self._regulatory_context = regulatory_context

    def advise(
        self,
        bias_metrics: dict[str, float],
        feature_contributions: dict[str, dict[str, Any]],
        intersectional_result: dict[str, Any] | None = None,
        model_type: str = "classification",
        n_samples: int = 1000,
    ) -> dict[str, Any]:
        """Generate a prioritized remediation roadmap from bias assessment results.

        Args:
            bias_metrics: Dict of metric_name → value from bias detection.
            feature_contributions: Bias attribution per feature from BiasAttributor.
            intersectional_result: Optional intersectional analysis result dict.
            model_type: Model type ('classification', 'regression', 'ranking').
            n_samples: Training set size (affects recommendation feasibility).

        Returns:
            Remediation roadmap dict with prioritized strategies and projections.
        """
        logger.info(
            "Generating remediation recommendations",
            n_metrics=len(bias_metrics),
            n_feature_contributions=len(feature_contributions),
            model_type=model_type,
        )

        # Assess bias pattern
        bias_pattern = self._assess_bias_pattern(
            bias_metrics=bias_metrics,
            feature_contributions=feature_contributions,
            intersectional_result=intersectional_result,
            n_samples=n_samples,
        )

        # Select applicable strategies
        applicable_strategies = self._select_strategies(bias_pattern)

        # Score and rank strategies
        scored_strategies = self._score_and_rank(
            strategies=applicable_strategies,
            bias_pattern=bias_pattern,
            bias_metrics=bias_metrics,
        )

        # Limit to max_recommendations
        top_strategies = scored_strategies[:self._max_recommendations]

        # Project before/after metrics
        projections = self._project_outcomes(
            strategies=top_strategies,
            bias_metrics=bias_metrics,
        )

        # Generate compliance context
        compliance_guidance = self._generate_compliance_guidance(
            bias_pattern=bias_pattern,
            top_strategies=top_strategies,
        )

        logger.info(
            "Remediation recommendations complete",
            n_recommendations=len(top_strategies),
            top_strategy=top_strategies[0]["strategy_id"] if top_strategies else None,
        )

        return {
            "remediation_method": "bias_pattern_analysis",
            "regulatory_context": self._regulatory_context,
            "model_type": model_type,
            "n_samples": n_samples,
            "bias_pattern": bias_pattern,
            "n_recommendations": len(top_strategies),
            "recommended_strategies": top_strategies,
            "projected_outcomes": projections,
            "implementation_roadmap": self._build_roadmap(top_strategies),
            "compliance_guidance": compliance_guidance,
            "total_estimated_days": sum(
                s.get("time_estimate_days", 0) for s in top_strategies
            ),
            "quick_win_available": any(
                s.get("implementation_difficulty") == "low" for s in top_strategies
            ),
        }

    def generate_feature_remediation(
        self,
        feature_name: str,
        bias_contribution: float,
        correlation_with_protected: float,
        variance_fraction: float,
    ) -> list[str]:
        """Generate feature-level remediation actions.

        Standalone method for per-feature bias reports.

        Args:
            feature_name: Name of the biased feature.
            bias_contribution: Feature's contribution to total bias gap.
            correlation_with_protected: Correlation with the protected attribute.
            variance_fraction: Feature's variance as fraction of total variance.

        Returns:
            List of feature-specific remediation action strings.
        """
        actions: list[str] = []

        if abs(correlation_with_protected) > 0.7:
            actions.append(
                f"CRITICAL: '{feature_name}' is a high-confidence proxy feature "
                f"(correlation={correlation_with_protected:.2f}). Remove or transform "
                f"via adversarial debiasing to eliminate proxy discrimination."
            )
        elif abs(correlation_with_protected) > 0.4:
            actions.append(
                f"'{feature_name}' shows moderate proxy discrimination risk "
                f"(correlation={correlation_with_protected:.2f}). Apply instance reweighting "
                f"or fairness-aware regularization targeting this feature."
            )

        if abs(bias_contribution) > 0.1:
            actions.append(
                f"'{feature_name}' contributes {bias_contribution:.3f} to total bias. "
                f"Collect additional training data from underrepresented groups "
                f"across this feature's value distribution."
            )

        if variance_fraction > 0.2:
            actions.append(
                f"'{feature_name}' accounts for {variance_fraction:.1%} of dataset variance. "
                f"Consider whether this feature's predictive power can be preserved "
                f"while reducing its discriminatory impact via fair representation learning."
            )

        if not actions:
            actions.append(
                f"Monitor '{feature_name}' in production for distributional drift "
                f"that could amplify its current low-level bias contribution."
            )

        return actions

    def _assess_bias_pattern(
        self,
        bias_metrics: dict[str, float],
        feature_contributions: dict[str, dict[str, Any]],
        intersectional_result: dict[str, Any] | None,
        n_samples: int,
    ) -> dict[str, Any]:
        """Assess the bias pattern from input metrics.

        Args:
            bias_metrics: Detected bias metrics.
            feature_contributions: Per-feature bias contributions.
            intersectional_result: Optional intersectional analysis data.
            n_samples: Training set size.

        Returns:
            Bias pattern characterization dict.
        """
        parity_gap = abs(bias_metrics.get("demographic_parity_difference", 0.0))
        odds_gap = abs(bias_metrics.get("equalized_odds_difference", 0.0))
        disparate_impact = bias_metrics.get("disparate_impact", 1.0)

        # Identify proxy features: high correlation with protected attribute
        proxy_features = [
            name for name, contrib in feature_contributions.items()
            if abs(contrib.get("correlation_with_protected", 0.0)) > 0.4
        ]

        # Identify small groups from intersectional result
        small_groups = False
        n_disparate_groups = 0
        intersectional_amplification = 1.0

        if intersectional_result:
            small_groups = any(
                g.get("n_samples", 0) < 100
                for g in intersectional_result.get("intersectional_groups", [])
            )
            n_disparate_groups = len(intersectional_result.get("disparate_groups", []))
            intersectional_amplification = intersectional_result.get("intersectional_amplification", 1.0)

        return {
            "high_parity_gap": parity_gap > 0.1,
            "high_odds_gap": odds_gap > 0.1,
            "disparate_impact_violation": disparate_impact < 0.8,
            "proxy_features": len(proxy_features) > 0,
            "proxy_feature_names": proxy_features,
            "small_groups": small_groups,
            "intersectional_amplification": intersectional_amplification > 1.5,
            "many_disparate_groups": n_disparate_groups > 3,
            "parity_gap": parity_gap,
            "odds_gap": odds_gap,
            "n_disparate_groups": n_disparate_groups,
            "n_samples": n_samples,
            "data_limited": n_samples < 1000,
        }

    def _select_strategies(self, bias_pattern: dict[str, Any]) -> list[str]:
        """Select applicable strategy IDs based on bias pattern.

        Args:
            bias_pattern: Bias pattern characterization dict.

        Returns:
            List of applicable strategy IDs without duplicates.
        """
        selected: list[str] = []

        for condition, strategies in STRATEGY_APPLICABILITY:
            if bias_pattern.get(condition, False):
                for strategy_id in strategies:
                    if strategy_id not in selected:
                        selected.append(strategy_id)

        # If no specific pattern matches, recommend general strategies
        if not selected:
            selected = ["threshold_adjustment", "instance_reweighting", "targeted_resampling"]

        # Filter by retraining constraint
        if self._require_no_retraining:
            no_retraining_strategies = {"threshold_adjustment", "calibrated_thresholds"}
            selected = [s for s in selected if s in no_retraining_strategies]
            if not selected:
                selected = ["threshold_adjustment"]

        return selected

    def _score_and_rank(
        self,
        strategies: list[str],
        bias_pattern: dict[str, Any],
        bias_metrics: dict[str, float],
    ) -> list[dict[str, Any]]:
        """Score and rank strategies by expected impact and feasibility.

        Args:
            strategies: Applicable strategy IDs.
            bias_pattern: Bias pattern characterization.
            bias_metrics: Current bias metric values.

        Returns:
            Sorted list of strategy dicts with scores and priority ranks.
        """
        parity_gap = bias_pattern.get("parity_gap", 0.0)
        odds_gap = bias_pattern.get("odds_gap", 0.0)

        scored: list[dict[str, Any]] = []

        for strategy_id in strategies:
            if strategy_id not in REMEDIATION_STRATEGIES:
                continue

            strategy = REMEDIATION_STRATEGIES[strategy_id]
            parity_reduction = strategy["expected_parity_reduction"]
            odds_reduction = strategy["expected_odds_reduction"]

            # Score: impact (70%) + feasibility (30%)
            impact_score = (parity_reduction * parity_gap + odds_reduction * odds_gap)
            difficulty_penalty = {"low": 0.0, "medium": 0.1, "high": 0.25}.get(
                strategy["implementation_difficulty"], 0.1
            )
            time_penalty = min(0.2, strategy["time_estimate_days"] / 100)

            if self._prefer_quick_wins:
                difficulty_penalty *= 2

            total_score = impact_score * 0.7 - difficulty_penalty * 0.2 - time_penalty * 0.1

            rationale = self._generate_rationale(strategy_id, bias_pattern, parity_gap, odds_gap)
            implementation_steps = self._generate_implementation_steps(strategy_id, bias_pattern)

            scored.append({
                "strategy_id": strategy_id,
                "name": strategy["name"],
                "category": strategy["category"],
                "description": strategy["description"],
                "score": round(total_score, 4),
                "rationale": rationale,
                "estimated_parity_reduction": parity_reduction,
                "estimated_odds_reduction": odds_reduction,
                "projected_parity_gap_after": round(max(0.0, parity_gap * (1 - parity_reduction)), 4),
                "projected_odds_gap_after": round(max(0.0, odds_gap * (1 - odds_reduction)), 4),
                "implementation_difficulty": strategy["implementation_difficulty"],
                "time_estimate_days": strategy["time_estimate_days"],
                "required_libraries": strategy["libraries"],
                "side_effects": strategy["side_effects"],
                "implementation_steps": implementation_steps,
            })

        scored.sort(key=lambda s: s["score"], reverse=True)

        for rank, strategy in enumerate(scored):
            strategy["priority"] = rank + 1

        return scored

    def _generate_rationale(
        self,
        strategy_id: str,
        bias_pattern: dict[str, Any],
        parity_gap: float,
        odds_gap: float,
    ) -> str:
        """Generate a concise rationale for recommending a strategy.

        Args:
            strategy_id: Strategy identifier.
            bias_pattern: Bias pattern characterization.
            parity_gap: Current parity gap.
            odds_gap: Current odds gap.

        Returns:
            Rationale string.
        """
        rationales = {
            "threshold_adjustment": (
                f"Fastest path to reducing parity gap ({parity_gap:.3f}) "
                f"without model retraining. Applicable immediately post-deployment."
            ),
            "calibrated_thresholds": (
                f"Equalized odds gap ({odds_gap:.3f}) indicates TPR/FPR imbalance. "
                f"Hardt et al. calibration directly targets this metric."
            ),
            "proxy_feature_removal": (
                f"High-correlation features detected: {', '.join(bias_pattern.get('proxy_feature_names', [])[:3])}. "
                f"Removing proxy features eliminates the structural bias source."
            ),
            "instance_reweighting": (
                f"Adjusting instance weights is low-risk and reversible. "
                f"Effective when group imbalance drives the parity gap."
            ),
            "targeted_resampling": (
                "Underrepresented intersectional groups detected. "
                "Oversampling increases representation in training data directly."
            ),
            "adversarial_debiasing": (
                "Proxy features and intersectional bias patterns suggest structural bias. "
                "Adversarial debiasing eliminates bias at the representation level."
            ),
            "fairness_constraints": (
                "Multiple disparate groups detected. Constraint-based training "
                "provides formal fairness guarantees across all groups simultaneously."
            ),
            "targeted_data_collection": (
                "Small group sizes limit the reliability of current disparity estimates. "
                "Additional representative data improves model and reduces uncertainty."
            ),
            "synthetic_augmentation": (
                "Targeted data collection is infeasible in the short term. "
                "Fairness-aware synthetic augmentation can improve group representation."
            ),
        }
        return rationales.get(strategy_id, f"Recommended for detected bias pattern in {strategy_id}.")

    def _generate_implementation_steps(
        self,
        strategy_id: str,
        bias_pattern: dict[str, Any],
    ) -> list[str]:
        """Generate step-by-step implementation guidance for a strategy.

        Args:
            strategy_id: Strategy identifier.
            bias_pattern: Bias pattern characterization.

        Returns:
            Ordered list of implementation step strings.
        """
        steps: dict[str, list[str]] = {
            "threshold_adjustment": [
                "1. Compute predicted probabilities from existing model on validation set",
                "2. Split validation set by protected group",
                "3. For each group, find threshold minimizing fairness objective (demographic parity or equalized odds)",
                "4. Validate thresholds on held-out test set",
                "5. Deploy group-specific thresholds in model serving layer",
                "6. Monitor parity gap weekly in production",
            ],
            "instance_reweighting": [
                "1. Compute demographic parity gap per group on training data",
                "2. Calculate sample weights inversely proportional to group selection rates",
                "3. Apply weights to model training loss (e.g., class_weight in sklearn)",
                "4. Retrain model with instance weights",
                "5. Validate new model against fairness metrics on held-out set",
                "6. Run aumos-fairness-suite full assessment before deployment",
            ],
            "proxy_feature_removal": [
                "1. Identify proxy features using BiasAttributor correlation analysis",
                f"2. Features to investigate: {', '.join(bias_pattern.get('proxy_feature_names', [])[:5])}",
                "3. Validate removal by checking accuracy drop is within tolerance",
                "4. Retrain model without proxy features",
                "5. Re-run fairness assessment to confirm bias reduction",
                "6. Update model card and technical documentation",
            ],
            "targeted_resampling": [
                "1. Identify underrepresented intersectional groups from IntersectionalAnalyzer output",
                "2. Compute target sample counts per group for balanced representation",
                "3. Apply SMOTE or RandomOverSampler from imbalanced-learn",
                "4. Validate that oversampled dataset preserves feature distributions",
                "5. Retrain model on balanced dataset",
                "6. Run fairness assessment to validate parity improvement",
            ],
            "adversarial_debiasing": [
                "1. Set up adversarial debiasing architecture: predictor + adversary",
                "2. Configure adversary to predict protected attribute from model logits",
                "3. Train jointly with minimax objective (maximize task accuracy, minimize adversary accuracy)",
                "4. Tune adversary strength hyperparameter via validation fairness metrics",
                "5. Evaluate on held-out test set for both accuracy and fairness",
                "6. Run full fairness suite assessment including intersectional analysis",
            ],
            "calibrated_thresholds": [
                "1. Collect predicted probabilities and labels by protected group",
                "2. Apply Hardt et al. (2016) post-processing: solve LP for optimal thresholds",
                "3. Validate equalized odds improvement on held-out set",
                "4. Deploy calibrated decision function in serving layer",
                "5. Confirm TPR/FPR parity across all groups in production",
            ],
        }
        return steps.get(strategy_id, [
            f"1. Review {strategy_id} documentation",
            "2. Apply strategy to training pipeline",
            "3. Validate fairness improvement on held-out set",
            "4. Deploy and monitor in production",
        ])

    def _project_outcomes(
        self,
        strategies: list[dict[str, Any]],
        bias_metrics: dict[str, float],
    ) -> dict[str, Any]:
        """Project metric values after applying recommended strategies.

        Args:
            strategies: Ranked strategy dicts.
            bias_metrics: Current bias metric values.

        Returns:
            Dict with before/after projections per metric.
        """
        parity_gap_before = abs(bias_metrics.get("demographic_parity_difference", 0.0))
        odds_gap_before = abs(bias_metrics.get("equalized_odds_difference", 0.0))

        if not strategies:
            return {
                "parity_gap_before": parity_gap_before,
                "parity_gap_after_top_strategy": parity_gap_before,
                "odds_gap_before": odds_gap_before,
                "odds_gap_after_top_strategy": odds_gap_before,
                "expected_compliance": False,
            }

        top = strategies[0]
        parity_after = top.get("projected_parity_gap_after", parity_gap_before)
        odds_after = top.get("projected_odds_gap_after", odds_gap_before)

        return {
            "parity_gap_before": round(parity_gap_before, 4),
            "parity_gap_after_top_strategy": round(parity_after, 4),
            "odds_gap_before": round(odds_gap_before, 4),
            "odds_gap_after_top_strategy": round(odds_after, 4),
            "expected_compliance_parity": parity_after < 0.1,
            "expected_compliance_odds": odds_after < 0.1,
            "expected_overall_compliance": parity_after < 0.1 and odds_after < 0.1,
            "top_strategy_applied": top["strategy_id"],
            "projection_confidence": "moderate",
            "projection_note": (
                "Projections are estimates based on literature benchmarks. "
                "Actual results depend on dataset characteristics and implementation quality. "
                "Validate all changes with a full aumos-fairness-suite assessment."
            ),
        }

    def _build_roadmap(self, strategies: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build a phased implementation roadmap.

        Args:
            strategies: Ranked strategy dicts.

        Returns:
            List of roadmap phases with strategies, timelines, and milestones.
        """
        phases: list[dict[str, Any]] = []
        cumulative_days = 0

        phase_groups = {"immediate": [], "short_term": [], "long_term": []}
        for strategy in strategies:
            days = strategy.get("time_estimate_days", 7)
            if days <= 3:
                phase_groups["immediate"].append(strategy)
            elif days <= 21:
                phase_groups["short_term"].append(strategy)
            else:
                phase_groups["long_term"].append(strategy)

        for phase_name, phase_strategies in phase_groups.items():
            if not phase_strategies:
                continue
            phase_days = max(s.get("time_estimate_days", 1) for s in phase_strategies)
            cumulative_days += phase_days
            phases.append({
                "phase": phase_name,
                "phase_label": {
                    "immediate": "Phase 1: Quick Wins (0-3 days)",
                    "short_term": "Phase 2: Short-Term Improvements (1-3 weeks)",
                    "long_term": "Phase 3: Structural Remediation (1-3 months)",
                }.get(phase_name, phase_name),
                "strategies": [s["strategy_id"] for s in phase_strategies],
                "estimated_days": phase_days,
                "cumulative_days": cumulative_days,
                "milestone": f"Fairness re-assessment at day {cumulative_days}",
            })

        return phases

    def _generate_compliance_guidance(
        self,
        bias_pattern: dict[str, Any],
        top_strategies: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate regulatory compliance context for the remediation plan.

        Args:
            bias_pattern: Detected bias pattern.
            top_strategies: Top recommended strategies.

        Returns:
            Compliance guidance dict with regulatory references.
        """
        urgency = "high" if bias_pattern.get("disparate_impact_violation") else "medium"
        if bias_pattern.get("high_parity_gap") and bias_pattern.get("high_odds_gap"):
            urgency = "critical"

        references = {
            "ecoa": [
                "Reg B (12 CFR Part 1002) — Equal Credit Opportunity Act Implementing Regulation",
                "CFPB Examination Procedures for ECOA compliance",
            ],
            "eu_ai_act": [
                "EU AI Act Article 9 — Risk Management System",
                "EU AI Act Article 10 — Data and Data Governance (bias examination required)",
                "EU AI Act Recital 44 — Non-discrimination principles",
            ],
            "general": [
                "NIST AI RMF 1.0 — Govern 1.6, Measure 2.5, Manage 4.1",
                "ISO/IEC TR 24027 — Bias in AI systems",
            ],
        }

        return {
            "regulatory_context": self._regulatory_context,
            "urgency": urgency,
            "regulatory_references": references.get(self._regulatory_context, references["general"]),
            "documentation_required": [
                "Bias assessment report with metrics before and after remediation",
                "Remediation action plan with timelines and owners",
                "Post-remediation validation results",
                "Ongoing monitoring plan with alert thresholds",
            ],
            "disclosure_recommendation": (
                "Prepare disclosure memo for legal/compliance team if parity gap > 0.2 "
                "or if system is used in credit, employment, or housing decisions."
            ),
        }


__all__ = ["RemediationAdvisor", "StrategyRecommendation", "REMEDIATION_STRATEGIES"]
