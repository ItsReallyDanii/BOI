"""
Smoke tests for v1.5 pipeline

Integration tests verifying the implemented pipeline components.
"""

import pytest
import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_import_pipeline():
    """Test that the main pipeline module can be imported."""
    import v1_5_pipeline
    assert hasattr(v1_5_pipeline, 'run_pipeline')
    assert hasattr(v1_5_pipeline, 'apply_bayesian_shrinkage')
    assert hasattr(v1_5_pipeline, 'calculate_ece')
    assert hasattr(v1_5_pipeline, 'PipelineConfig')
    assert hasattr(v1_5_pipeline, 'SegmentedCalibrator')
    assert hasattr(v1_5_pipeline, 'PortfolioOptimizer')
    assert hasattr(v1_5_pipeline, 'brier_score')


def test_config_dataclass():
    """Test that PipelineConfig dataclass is properly defined."""
    from v1_5_pipeline import PipelineConfig

    config = PipelineConfig()
    assert config.bayes_c == 50
    assert config.quality_cap == 0.75
    assert config.ece_blacklist_threshold == 0.05
    assert config.max_kelly_cap == 0.25
    assert config.staleness_days == 7
    assert config.min_segment_samples == 50
    assert config.min_isotonic_samples == 300
    assert "power_2_pick" in config.contest_payout_multipliers
    assert config.contest_payout_multipliers["power_2_pick"] == 3.0


def test_bayesian_shrinkage_formula():
    """Test Bayesian shrinkage formula: p_shrunk = (n/(n+C))*p + (C/(n+C))*0.5"""
    from v1_5_pipeline import apply_bayesian_shrinkage

    # Low sample -> strong shrinkage toward 0.5
    result = apply_bayesian_shrinkage(0.8, sample_size=10, c=50)
    expected = (10 / 60) * 0.8 + (50 / 60) * 0.5
    assert abs(result - expected) < 1e-10

    # High sample -> minimal shrinkage
    result = apply_bayesian_shrinkage(0.8, sample_size=1000, c=50)
    expected = (1000 / 1050) * 0.8 + (50 / 1050) * 0.5
    assert abs(result - expected) < 1e-10

    # p=0.5 should stay at 0.5 regardless of n
    result = apply_bayesian_shrinkage(0.5, sample_size=10, c=50)
    assert abs(result - 0.5) < 1e-10


def test_ece_perfect_calibration():
    """ECE should be 0 for perfectly calibrated predictions."""
    from v1_5_pipeline import calculate_ece

    predicted = np.array([0.0, 0.0, 1.0, 1.0])
    actual = np.array([0, 0, 1, 1])
    ece = calculate_ece(predicted, actual)
    assert ece < 0.01


def test_ece_empty_input():
    """ECE should return 1.0 for empty arrays."""
    from v1_5_pipeline import calculate_ece

    ece = calculate_ece(np.array([]), np.array([]))
    assert ece == 1.0


def test_brier_score():
    """Test Brier score computation."""
    from v1_5_pipeline import brier_score

    predicted = np.array([1.0, 0.0])
    actual = np.array([1, 0])
    assert brier_score(predicted, actual) == 0.0

    predicted = np.array([0.0, 1.0])
    actual = np.array([1, 0])
    assert brier_score(predicted, actual) == 1.0


def test_segmented_calibrator_fit_and_resolve():
    """Test that calibrators fit and fallback resolution works."""
    from v1_5_pipeline import SegmentedCalibrator, PipelineConfig

    config = PipelineConfig()
    cal = SegmentedCalibrator(config)

    rng = np.random.RandomState(99)
    n = 400
    preds = np.clip(rng.beta(2, 2, size=n), 0.01, 0.99)
    outcomes = rng.binomial(1, preds)

    now = datetime(2026, 2, 8, 12, 0, 0)
    cal.fit_segment("NBA_PTS", preds, outcomes, retrained_date=now - timedelta(days=1))

    # Should resolve to specific segment
    calibrator, key, stale = cal.resolve("NBA", "PTS", now=now)
    assert key == "NBA_PTS"
    assert calibrator is not None
    assert stale is False


def test_segmented_calibrator_fallback_to_generic():
    """If specific segment has no calibrator, fall back to sport generic."""
    from v1_5_pipeline import SegmentedCalibrator, PipelineConfig

    config = PipelineConfig()
    cal = SegmentedCalibrator(config)

    rng = np.random.RandomState(99)
    now = datetime(2026, 2, 8, 12, 0, 0)

    # Only fit the generic segment
    n = 400
    preds = np.clip(rng.beta(2, 2, size=n), 0.01, 0.99)
    outcomes = rng.binomial(1, preds)
    cal.fit_segment("NBA_ALL", preds, outcomes, retrained_date=now - timedelta(days=1))

    # Resolve for NBA_AST which has no specific calibrator
    calibrator, key, stale = cal.resolve("NBA", "AST", now=now)
    assert key == "NBA_ALL"
    assert stale is False


def test_segmented_calibrator_staleness():
    """Stale segments should be skipped in fallback resolution."""
    from v1_5_pipeline import SegmentedCalibrator, PipelineConfig

    config = PipelineConfig()
    cal = SegmentedCalibrator(config)

    rng = np.random.RandomState(99)
    now = datetime(2026, 2, 8, 12, 0, 0)

    # Fit segment but mark it as old (10 days ago)
    n = 400
    preds = np.clip(rng.beta(2, 2, size=n), 0.01, 0.99)
    outcomes = rng.binomial(1, preds)
    cal.fit_segment("NBA_PTS", preds, outcomes, retrained_date=now - timedelta(days=10))

    # Fit GLOBAL as fresh
    preds2 = np.clip(rng.beta(2, 2, size=800), 0.01, 0.99)
    outcomes2 = rng.binomial(1, preds2)
    cal.fit_segment("GLOBAL", preds2, outcomes2, retrained_date=now - timedelta(days=1))

    # Should skip stale NBA_PTS and fall back to GLOBAL
    calibrator, key, stale = cal.resolve("NBA", "PTS", now=now)
    assert key == "GLOBAL"


def test_segmented_calibrator_insufficient_samples():
    """Segments with < min_segment_samples should not store a calibrator."""
    from v1_5_pipeline import SegmentedCalibrator, PipelineConfig

    config = PipelineConfig()
    cal = SegmentedCalibrator(config)

    rng = np.random.RandomState(99)
    now = datetime(2026, 2, 8, 12, 0, 0)

    preds = rng.beta(2, 2, size=30)  # below 50
    outcomes = rng.binomial(1, np.clip(preds, 0.01, 0.99))
    cal.fit_segment("MLB_HR", preds, outcomes, retrained_date=now)

    assert "MLB_HR" not in cal._calibrators


def test_platt_scaling_for_small_train():
    """Train n < 300 should use Platt scaling (LogisticRegression)."""
    from v1_5_pipeline import SegmentedCalibrator, PipelineConfig
    from sklearn.linear_model import LogisticRegression

    config = PipelineConfig()
    cal = SegmentedCalibrator(config)

    rng = np.random.RandomState(99)
    now = datetime(2026, 2, 8, 12, 0, 0)

    # 120 samples: 80% train = 96 < 300 -> Platt
    n = 120
    preds = np.clip(rng.beta(2, 2, size=n), 0.01, 0.99)
    outcomes = rng.binomial(1, preds)
    cal.fit_segment("NBA_REB", preds, outcomes, retrained_date=now)

    assert "NBA_REB" in cal._calibrators
    assert isinstance(cal._calibrators["NBA_REB"], LogisticRegression)


def test_isotonic_for_large_train():
    """Train n >= 300 should use IsotonicRegression."""
    from v1_5_pipeline import SegmentedCalibrator, PipelineConfig
    from sklearn.isotonic import IsotonicRegression

    config = PipelineConfig()
    cal = SegmentedCalibrator(config)

    rng = np.random.RandomState(99)
    now = datetime(2026, 2, 8, 12, 0, 0)

    # 500 samples: 80% train = 400 >= 300 -> isotonic
    n = 500
    preds = np.clip(rng.beta(2, 2, size=n), 0.01, 0.99)
    outcomes = rng.binomial(1, preds)
    cal.fit_segment("NBA_PTS", preds, outcomes, retrained_date=now)

    assert "NBA_PTS" in cal._calibrators
    assert isinstance(cal._calibrators["NBA_PTS"], IsotonicRegression)


def test_payout_multiplier_lookup():
    """Test contest payout multiplier lookup."""
    from v1_5_pipeline import payout_multiplier, PipelineConfig

    config = PipelineConfig()
    assert payout_multiplier("power_2_pick", config) == 3.0
    assert payout_multiplier("flex_2_pick", config) == 2.0

    with pytest.raises(ValueError):
        payout_multiplier("nonexistent_type", config)


def test_process_leg_probability_quality_cap():
    """Quality cap should limit prob when news_confidence_low is True."""
    from v1_5_pipeline import process_leg_probability, PipelineConfig, SegmentedCalibrator

    config = PipelineConfig()
    seg_cal = SegmentedCalibrator(config)

    # With cap: 0.90 should be capped to 0.75 before shrinkage
    p_capped = process_leg_probability(
        raw_prob=0.90, sample_size=500, news_confidence_low=True,
        calibrator=None, seg_calibrator=seg_cal, config=config,
    )
    # Without cap
    p_uncapped = process_leg_probability(
        raw_prob=0.90, sample_size=500, news_confidence_low=False,
        calibrator=None, seg_calibrator=seg_cal, config=config,
    )
    assert p_capped < p_uncapped


def test_process_leg_probability_clip():
    """Final probability should always be in [0.01, 0.99]."""
    from v1_5_pipeline import process_leg_probability, PipelineConfig, SegmentedCalibrator

    config = PipelineConfig()
    seg_cal = SegmentedCalibrator(config)

    p = process_leg_probability(
        raw_prob=0.001, sample_size=5, news_confidence_low=False,
        calibrator=None, seg_calibrator=seg_cal, config=config,
    )
    assert p >= 0.01
    assert p <= 0.99


def test_portfolio_optimizer_rejects_same_game():
    """Optimizer should reject pairs from the same game_id."""
    import pandas as pd
    from v1_5_pipeline import PortfolioOptimizer, PipelineConfig, SegmentedCalibrator

    config = PipelineConfig()
    seg_cal = SegmentedCalibrator(config)

    legs = pd.DataFrame([
        {"leg_id": "A", "game_id": "G1", "player_id": "P1", "sport": "NBA",
         "prop_type": "PTS", "p_shrunk": 0.7, "edge": 0.3,
         "quality_flag": True, "segment_key": "NBA_PTS", "stale": False},
        {"leg_id": "B", "game_id": "G1", "player_id": "P2", "sport": "NBA",
         "prop_type": "REB", "p_shrunk": 0.65, "edge": 0.2,
         "quality_flag": True, "segment_key": "NBA_REB", "stale": False},
    ])

    opt = PortfolioOptimizer(config)
    result = opt.optimize(legs, "power_2_pick", seg_cal)
    assert result is None


def test_pipeline_runs_without_error():
    """Full pipeline demo should execute without exceptions."""
    from v1_5_pipeline import run_pipeline

    # Should not raise
    run_pipeline()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
