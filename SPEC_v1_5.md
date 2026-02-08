# v1.5 Pipeline Specification

## Overview

This document defines the frozen specification for the v1.5 quant pick'em pipeline. The pipeline transforms raw model probabilities into calibrated predictions and identifies optimal 2-leg bet pairs.

## Calibration Fallback Chain

Models are resolved using a hierarchical fallback strategy:

```
Specific Segment → Sport Generic → Global Default
```

**Example**: For an NBA points prop at home:
1. Try `NBA_PTS_Home` (specific segment)
2. Fall back to `NBA_ALL` (sport generic)
3. Fall back to `GLOBAL` (cross-sport)

Each level requires `min_segment_samples` to be considered valid.

## Probability Transformation Pipeline

Raw model probabilities undergo sequential transformations:

```
Raw Prob → Quality Cap → Bayesian Shrinkage → Calibration → Clip [0.01, 0.99]
```

### 1. Quality Cap
```python
if p > quality_cap:
    p_capped = quality_cap
else:
    p_capped = p
```

### 2. Bayesian Shrinkage

Formula:
```
p_shrunk = (n/(n+C)) * p + (C/(n+C)) * 0.5
```

Where:
- `n` = sample size for the model
- `C` = 50 (shrinkage constant from `settings.yaml`)
- `0.5` = prior (50% base rate)

**Effect**: Low-sample models are pulled toward 50%, high-sample models remain closer to original.

### 3. Isotonic Calibration

Apply isotonic regression if segment has ≥ `min_isotonic_samples` historical predictions.

### 4. Final Clipping

```python
p_final = clip(p_calibrated, 0.01, 0.99)
```

Prevents degenerate probabilities that break EV calculations.

## Expected Calibration Error (ECE)

ECE measures calibration quality by binning predictions and comparing to actual outcomes.

**Calculation**:
```
ECE = Σ (|bin_avg_predicted - bin_avg_actual| * bin_weight)
```

**Blacklist Trigger**: If ECE > `ece_blacklist_threshold` (0.05), the model is excluded from consideration.

## 2-Leg Optimizer

The pipeline constructs all-pairs combinations of propositions and selects the best by expected value.

### Pairing Logic
- Generate all combinations: `combinations(props, 2)`
- For each pair:
  - Calculate joint probability: `P(A ∩ B) = P(A) × P(B)` (assumes independence)
  - Lookup payout multiplier from `payouts.yaml`
  - Calculate EV: `(joint_prob × payout) - 1`
  - Calculate Kelly fraction: `(prob × payout - 1) / (payout - 1)`
  - Cap Kelly at `max_kelly_cap`

### Selection Criteria
Return the pair with:
- Maximum EV > 0
- Both legs pass calibration checks
- Neither leg violates no-bet rules

## No-Bet Conditions

A proposition is excluded if ANY of these apply:

1. **Stale Data**: `(current_time - timestamp) > staleness_days`
2. **High ECE**: Model's ECE > `ece_blacklist_threshold`
3. **Insufficient Samples**: Segment samples < `min_segment_samples` at all fallback levels
4. **Negative EV**: Final expected value ≤ 0
5. **Probability Bounds**: Calibrated probability not in [0.01, 0.99]
6. **Missing Calibration**: No valid calibrator in fallback chain

## Configuration Parameters

From `configs/settings.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bayes_c` | 50 | Bayesian shrinkage constant |
| `quality_cap` | 0.75 | Max probability before shrinkage |
| `ece_blacklist_threshold` | 0.05 | ECE cutoff for model exclusion |
| `staleness_days` | 7 | Max data age in days |
| `min_segment_samples` | 50 | Min samples for segment calibrator |
| `min_isotonic_samples` | 300 | Min samples for isotonic regression |
| `max_kelly_cap` | 0.25 | Max Kelly bet size (fraction) |

## Implementation Notes

- All probability calculations use standard IEEE 754 floating point
- Timestamps must be ISO 8601 format
- Segment keys are case-sensitive
- Missing ECE scores are treated as failing the ECE check
- Ties in EV are broken by lower combined variance (if implemented)

---

**Version**: 1.5 (Frozen)  
**Last Updated**: 2026-02-08
