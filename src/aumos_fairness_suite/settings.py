"""Service-specific settings extending AumOS base config."""

from pydantic_settings import SettingsConfigDict

from aumos_common.config import AumOSSettings


class Settings(AumOSSettings):
    """Configuration for aumos-fairness-suite.

    All environment variables use the AUMOS_FAIRNESS_ prefix.
    Standard AumOS settings (database, kafka, log level) are inherited from AumOSSettings.
    """

    service_name: str = "aumos-fairness-suite"

    # --- Fairness metric thresholds ---
    # Disparate impact ratio: pass if value >= this threshold (the "4/5 rule")
    disparate_impact_threshold: float = 0.8
    # Absolute difference metrics: pass if abs(value) <= this threshold
    parity_difference_threshold: float = 0.1
    # Ratio metrics: pass if value >= this threshold
    parity_ratio_threshold: float = 0.8
    # Theil index: pass if value < this threshold
    theil_index_threshold: float = 0.1

    # --- Synthetic bias thresholds ---
    # KL divergence for group distribution shift: pass if value < threshold
    kl_divergence_threshold: float = 0.05
    # Label rate disparity between real and synthetic: pass if abs(diff) < threshold
    label_rate_disparity_threshold: float = 0.05
    # Amplification factor (synthetic / real bias): pass if factor < threshold
    amplification_threshold: float = 1.2

    # --- Monitoring ---
    default_monitor_cron: str = "0 */6 * * *"
    monitor_window_size: int = 10000

    # --- Upstream service URLs ---
    governance_engine_url: str = "http://localhost:8016"
    mlops_url: str = "http://localhost:8008"

    # --- Report defaults ---
    default_report_frameworks: list[str] = ["ECOA", "EU_AI_ACT"]

    model_config = SettingsConfigDict(env_prefix="AUMOS_FAIRNESS_")
