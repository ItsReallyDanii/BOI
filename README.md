# v1.5 Quant Pick'em Pipeline

## Objective

This pipeline implements a quantitative analysis system for sports betting pick'em contests. It processes proposition bets, applies Bayesian probability adjustments, calibrates model predictions, and identifies optimal 2-leg bet combinations based on expected value calculations.

**This is a process/analysis-only pipeline** - it does not execute bets or connect to external APIs. All operations are local and deterministic.

## Local-Only Constraints

- **No network calls**: All data must be provided as local CSV/JSON files
- **No external APIs**: No live odds feeds, no sportsbook APIs
- **Reproducible**: Same input data produces same output recommendations
- **Offline-first**: Works in air-gapped environments

## Repository Structure

```
.
├── README.md                 # This file
├── SPEC_v1_5.md             # Technical specification
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
├── PROMPT_OPUS_ONE_SHOT.txt # Claude prompt template
├── src/
│   ├── __init__.py
│   └── v1_5_pipeline.py    # Main pipeline implementation
├── data/
│   ├── raw/                # Input data files (user-provided)
│   └── processed/          # Pipeline outputs
├── configs/
│   ├── payouts.yaml        # Contest payout multipliers
│   └── settings.yaml       # Pipeline hyperparameters
├── logs/
│   └── .gitkeep
└── tests/
    ├── __init__.py
    └── test_smoke.py       # Basic integration tests
```

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline (after implementing)
python src/v1_5_pipeline.py

# Run tests
pytest tests/
```

## Input Schema

Place input files in `data/raw/`. Expected columns:

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| `prop_id` | str | Unique proposition identifier | ✓ |
| `player_name` | str | Player name | ✓ |
| `stat_type` | str | Stat category (e.g., "PTS", "REB") | ✓ |
| `line` | float | Over/under line value | ✓ |
| `model_prob` | float | Raw model probability (0-1) | ✓ |
| `sport` | str | Sport (e.g., "NBA", "NFL") | ✓ |
| `segment` | str | Segment key (e.g., "NBA_PTS_Home") | ✓ |
| `sample_size` | int | Model training samples | ✓ |
| `timestamp` | str | Data generation time (ISO 8601) | ✓ |
| `ece_score` | float | Model calibration error | Optional |

## Output Schema

Pipeline writes to `data/processed/recommendations.csv`:

| Column | Type | Description |
|--------|------|-------------|
| `leg1_prop_id` | str | First proposition ID |
| `leg2_prop_id` | str | Second proposition ID |
| `leg1_prob` | float | Calibrated probability (leg 1) |
| `leg2_prob` | float | Calibrated probability (leg 2) |
| `combined_prob` | float | Joint probability (leg1 × leg2) |
| `payout_multiplier` | float | Contest payout ratio |
| `expected_value` | float | EV = combined_prob × payout - 1 |
| `kelly_fraction` | float | Suggested bet size (0-max_kelly_cap) |
| `reason` | str | Selection rationale |

## No-Bet Rules

The pipeline will **not recommend** a bet if any of these conditions are met:

1. **Stale data**: Proposition timestamp older than `staleness_days`
2. **Blacklisted model**: Model ECE > `ece_blacklist_threshold`
3. **Insufficient calibration data**: Segment samples < `min_segment_samples`
4. **Negative EV**: Expected value ≤ 0 after all adjustments
5. **Probability extremes**: Final probability outside [0.01, 0.99] bounds
6. **Missing fallback**: No valid calibration model in fallback chain

---

**Status**: Scaffold created. Implementation pending.