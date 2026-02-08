"""
v1.5 Quant Pick'em Pipeline

Production-ready implementation of the segmented calibration, Bayesian shrinkage,
portfolio optimization, and Kelly sizing pipeline.

See SPEC_v1_5.md for detailed technical specification.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from itertools import combinations
from datetime import datetime, timedelta
import warnings

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


# ============================================================================
# Configuration
# ============================================================================

CONTEST_PAYOUT_MULTIPLIERS: Dict[str, float] = {
    "power_2_pick": 3.0,
    "power_3_pick": 6.0,
    "power_4_pick": 10.0,
    "power_5_pick": 20.0,
    "flex_2_pick": 2.0,
    "flex_3_pick": 2.25,
    "flex_4_pick": 5.0,
    "flex_5_pick": 10.0,
}


@dataclass
class PipelineConfig:
    """Pipeline configuration settings loaded from configs/settings.yaml"""
    bayes_c: int = 50
    quality_cap: float = 0.75
    ece_blacklist_threshold: float = 0.05
    staleness_days: int = 7
    min_segment_samples: int = 50
    min_isotonic_samples: int = 300
    max_kelly_cap: float = 0.25
    contest_payout_multipliers: Dict[str, float] = field(
        default_factory=lambda: dict(CONTEST_PAYOUT_MULTIPLIERS)
    )


def payout_multiplier(contest_type: str, config: PipelineConfig) -> float:
    """Look up payout multiplier for a contest type."""
    if contest_type not in config.contest_payout_multipliers:
        raise ValueError(
            f"Unknown contest_type '{contest_type}'. "
            f"Valid types: {list(config.contest_payout_multipliers.keys())}"
        )
    return config.contest_payout_multipliers[contest_type]


# ============================================================================
# Bayesian Shrinkage Helper
# ============================================================================

def apply_bayesian_shrinkage(prob: float, sample_size: int, c: int = 50) -> float:
    """
    Apply Bayesian shrinkage toward 0.5 prior.

    Formula: p_shrunk = (n / (n + C)) * p + (C / (n + C)) * 0.5

    Low-sample models are pulled strongly toward 0.5;
    high-sample models remain close to their raw probability.
    """
    n = float(sample_size)
    C = float(c)
    return (n / (n + C)) * prob + (C / (n + C)) * 0.5


# ============================================================================
# Expected Calibration Error (ECE)
# ============================================================================

def calculate_ece(
    predicted: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Calculate Expected Calibration Error.

    ECE = sum over bins of (|avg_predicted - avg_actual| * bin_weight)
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)

    if len(predicted) == 0:
        return 1.0  # degenerate: treat as worst calibration

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(predicted)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        if lo == bin_edges[-2]:
            mask = (predicted >= lo) & (predicted <= hi)
        else:
            mask = (predicted >= lo) & (predicted < hi)
        count = mask.sum()
        if count == 0:
            continue
        avg_pred = predicted[mask].mean()
        avg_actual = actual[mask].mean()
        ece += (count / total) * abs(avg_pred - avg_actual)

    return float(ece)


def brier_score(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Brier score: mean squared error between predicted probs and outcomes."""
    predicted = np.asarray(predicted, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if len(predicted) == 0:
        return 1.0
    return float(np.mean((predicted - actual) ** 2))


# ============================================================================
# SegmentedCalibrator
# ============================================================================

class SegmentedCalibrator:
    """
    Manages per-segment calibration models with hierarchical fallback.

    Segment key = sport + "_" + prop_type  (e.g. "NBA_PTS").
    Fallback chain: Specific Segment -> Sport Generic -> Global.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        # segment_key -> fitted calibrator
        self._calibrators: Dict[str, object] = {}
        # segment_key -> sample count used for training
        self._segment_n: Dict[str, int] = {}
        # segment_key -> last retrained datetime
        self._retrained_dates: Dict[str, datetime] = {}
        # segment_key -> ECE on test split
        self._segment_ece: Dict[str, float] = {}
        # segment_key -> Brier score on test split
        self._segment_brier: Dict[str, float] = {}
        # set of blacklisted segment keys
        self.blacklisted: set = set()

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_segment(
        self,
        segment_key: str,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        retrained_date: Optional[datetime] = None,
    ) -> None:
        """
        Fit a calibrator for *segment_key* using chronological train/test split.

        - Oldest 80 % train, newest 20 % test, no shuffle.
        - If train n < 300 -> Platt scaling (LogisticRegression).
        - If train n >= 300 -> IsotonicRegression.
        - Single-class edge case handled safely.
        """
        predictions = np.asarray(predictions, dtype=np.float64)
        outcomes = np.asarray(outcomes, dtype=np.float64)
        n = len(predictions)

        self._segment_n[segment_key] = n
        self._retrained_dates[segment_key] = retrained_date or datetime.utcnow()

        if n < self.config.min_segment_samples:
            # Not enough data; do NOT store a calibrator for this segment.
            return

        split = int(n * 0.8)
        train_pred, test_pred = predictions[:split], predictions[split:]
        train_out, test_out = outcomes[:split], outcomes[split:]

        # Guard: single-class outcomes in train set
        unique_classes = np.unique(train_out)
        if len(unique_classes) < 2:
            # Cannot fit a meaningful calibrator; store identity-like pass-through
            warnings.warn(
                f"Segment '{segment_key}' has single-class training data. "
                "Storing no calibrator."
            )
            # Still compute ECE/Brier on test set for audit
            if len(test_pred) > 0 and len(np.unique(test_out)) >= 1:
                self._segment_ece[segment_key] = calculate_ece(test_pred, test_out)
                self._segment_brier[segment_key] = brier_score(test_pred, test_out)
            return

        train_n = len(train_pred)

        if train_n < self.config.min_isotonic_samples:
            # Platt scaling via logistic regression
            calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
            calibrator.fit(train_pred.reshape(-1, 1), train_out)
            cal_test = calibrator.predict_proba(test_pred.reshape(-1, 1))[:, 1]
        else:
            # Isotonic regression
            calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            calibrator.fit(train_pred, train_out)
            cal_test = calibrator.predict(test_pred)

        self._calibrators[segment_key] = calibrator

        # Compute per-segment ECE and Brier on test split
        self._segment_ece[segment_key] = calculate_ece(cal_test, test_out)
        self._segment_brier[segment_key] = brier_score(cal_test, test_out)

        # Blacklist if ECE exceeds threshold
        if self._segment_ece[segment_key] > self.config.ece_blacklist_threshold:
            self.blacklisted.add(segment_key)

    # ------------------------------------------------------------------
    # Staleness check
    # ------------------------------------------------------------------

    def _is_stale(self, segment_key: str, now: Optional[datetime] = None) -> bool:
        now = now or datetime.utcnow()
        retrained = self._retrained_dates.get(segment_key)
        if retrained is None:
            return True
        return (now - retrained) > timedelta(days=self.config.staleness_days)

    # ------------------------------------------------------------------
    # Fallback resolution
    # ------------------------------------------------------------------

    def _segment_usable(self, key: str, now: Optional[datetime] = None) -> bool:
        """A segment is usable if it has a calibrator, enough samples, and is not stale."""
        if key not in self._calibrators:
            return False
        if self._segment_n.get(key, 0) < self.config.min_segment_samples:
            return False
        if self._is_stale(key, now):
            return False
        return True

    def resolve(
        self,
        sport: str,
        prop_type: str,
        now: Optional[datetime] = None,
    ) -> Tuple[Optional[object], str, bool]:
        """
        Resolve calibrator via fallback chain.

        Returns (calibrator_or_None, resolved_segment_key, is_stale_flag).
        """
        specific = f"{sport}_{prop_type}"
        generic = f"{sport}_ALL"
        global_key = "GLOBAL"

        for key in (specific, generic, global_key):
            if self._segment_usable(key, now):
                return self._calibrators[key], key, False

        # No usable calibrator found in chain
        return None, "NONE", True

    # ------------------------------------------------------------------
    # Transform a single probability through a resolved calibrator
    # ------------------------------------------------------------------

    def calibrate(self, prob: float, calibrator: object) -> float:
        """Apply a fitted calibrator to a single probability value."""
        if isinstance(calibrator, IsotonicRegression):
            return float(calibrator.predict(np.array([prob]))[0])
        elif isinstance(calibrator, LogisticRegression):
            return float(
                calibrator.predict_proba(np.array([[prob]]))[:, 1][0]
            )
        # Fallback: return raw
        return prob

    # ------------------------------------------------------------------
    # Audit report
    # ------------------------------------------------------------------

    def audit(self) -> pd.DataFrame:
        """Return a dataframe summarising per-segment calibration health."""
        rows = []
        now = datetime.utcnow()
        all_keys = set(self._segment_n.keys())
        for key in sorted(all_keys):
            rows.append({
                "segment": key,
                "n_samples": self._segment_n.get(key, 0),
                "has_calibrator": key in self._calibrators,
                "retrained_date": self._retrained_dates.get(key),
                "is_stale": self._is_stale(key, now),
                "ece": self._segment_ece.get(key, np.nan),
                "brier": self._segment_brier.get(key, np.nan),
                "blacklisted": key in self.blacklisted,
            })
        return pd.DataFrame(rows)


# ============================================================================
# Probability Processing (per-leg)
# ============================================================================

def process_leg_probability(
    raw_prob: float,
    sample_size: int,
    news_confidence_low: bool,
    calibrator: Optional[object],
    seg_calibrator: SegmentedCalibrator,
    config: PipelineConfig,
) -> float:
    """
    Full probability processing for a single leg.

    Order (MANDATORY):
      1. raw_model_prob
      2. quality cap if low news confidence (cap at 0.75)
      3. Bayesian shrinkage
      4. clip to [0.01, 0.99]
    """
    p = raw_prob

    # Step 1: quality cap
    if news_confidence_low:
        p = min(p, config.quality_cap)

    # Step 2: Bayesian shrinkage
    p = apply_bayesian_shrinkage(p, sample_size, c=config.bayes_c)

    # Step 3: clip
    p = float(np.clip(p, 0.01, 0.99))

    return p


# ============================================================================
# Portfolio Optimizer (2-leg, all-pairs brute force)
# ============================================================================

class PortfolioOptimizer:
    """Select optimal 2-leg pair via brute-force EV maximization."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def optimize(
        self,
        legs: pd.DataFrame,
        contest_type: str,
        seg_calibrator: SegmentedCalibrator,
    ) -> Optional[Dict]:
        """
        Find the best 2-leg pair.

        legs DataFrame expected columns:
            leg_id, game_id, player_id, sport, prop_type,
            p_shrunk, edge, quality_flag, segment_key

        Returns dict with pair info or None (NO BET).
        """
        mult = payout_multiplier(contest_type, self.config)

        # Filter candidate legs
        candidates = legs[
            (legs["edge"] > 0)
            & (legs["quality_flag"] == True)  # noqa: E712
            & (~legs["segment_key"].isin(seg_calibrator.blacklisted))
            & (legs["stale"] == False)  # noqa: E712
        ].copy()

        if len(candidates) < 2:
            return None

        best_pair: Optional[Dict] = None
        best_ev = 0.0

        indices = candidates.index.tolist()
        for i, j in combinations(indices, 2):
            row_a = candidates.loc[i]
            row_b = candidates.loc[j]

            # Correlation rejects
            if row_a["game_id"] == row_b["game_id"]:
                continue
            if row_a["player_id"] == row_b["player_id"]:
                continue

            p_joint = row_a["p_shrunk"] * row_b["p_shrunk"]
            ev = (p_joint * mult) - 1.0

            if ev > best_ev:
                best_ev = ev
                best_pair = {
                    "leg1": row_a.to_dict(),
                    "leg2": row_b.to_dict(),
                    "joint_prob": p_joint,
                    "payout_multiplier": mult,
                    "ev": ev,
                    "contest_type": contest_type,
                }

        if best_pair is None or best_pair["ev"] <= 0:
            return None

        # Kelly sizing for the pair
        b = mult - 1.0
        p = best_pair["joint_prob"]
        if b > 0:
            f_star = ((b * p) - (1.0 - p)) / b
        else:
            f_star = 0.0
        f_clamped = max(0.0, min(f_star, self.config.max_kelly_cap))

        best_pair["kelly_fraction"] = f_clamped

        return best_pair


# ============================================================================
# Demo: run_pipeline with mock data
# ============================================================================

def _build_mock_data(
    config: PipelineConfig,
    now: datetime,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[np.ndarray, np.ndarray, datetime]]]:
    """
    Build deterministic mock data for the demo run.

    Returns:
        (legs_df, historical_dict)
        historical_dict maps segment_key -> (predictions, outcomes, retrained_date)
    """
    rng = np.random.RandomState(42)

    # -- Historical calibration data per segment --
    def _make_history(n: int, bias: float, days_ago: int):
        preds = np.clip(rng.beta(2, 2, size=n) + bias, 0, 1)
        noise = rng.binomial(1, np.clip(preds, 0.01, 0.99))
        retrained = now - timedelta(days=days_ago)
        return preds, noise, retrained

    historical: Dict[str, Tuple[np.ndarray, np.ndarray, datetime]] = {
        "NBA_PTS":  _make_history(400, 0.05, 1),
        "NBA_REB":  _make_history(120, 0.0, 2),
        "NBA_ALL":  _make_history(500, 0.02, 1),
        "NFL_PASS": _make_history(350, 0.03, 3),
        "NFL_ALL":  _make_history(450, 0.01, 2),
        "MLB_HR":   _make_history(40, 0.0, 1),   # below min_segment_samples
        "GLOBAL":   _make_history(800, 0.0, 1),
        "STALE_SEG": _make_history(300, 0.0, 10),  # stale (>7 days)
    }

    # -- Today's legs --
    legs_raw = [
        {
            "leg_id": "L001", "game_id": "G100", "player_id": "P10",
            "sport": "NBA", "prop_type": "PTS",
            "raw_prob": 0.72, "sample_size": 200,
            "news_confidence_low": False,
        },
        {
            "leg_id": "L002", "game_id": "G101", "player_id": "P11",
            "sport": "NBA", "prop_type": "REB",
            "raw_prob": 0.65, "sample_size": 60,
            "news_confidence_low": True,
        },
        {
            "leg_id": "L003", "game_id": "G200", "player_id": "P20",
            "sport": "NFL", "prop_type": "PASS",
            "raw_prob": 0.80, "sample_size": 350,
            "news_confidence_low": False,
        },
        {
            "leg_id": "L004", "game_id": "G200", "player_id": "P21",
            "sport": "NFL", "prop_type": "PASS",
            "raw_prob": 0.68, "sample_size": 150,
            "news_confidence_low": False,
        },
        {
            "leg_id": "L005", "game_id": "G300", "player_id": "P30",
            "sport": "MLB", "prop_type": "HR",
            "raw_prob": 0.55, "sample_size": 30,
            "news_confidence_low": True,
        },
        {
            "leg_id": "L006", "game_id": "G301", "player_id": "P31",
            "sport": "NBA", "prop_type": "PTS",
            "raw_prob": 0.78, "sample_size": 400,
            "news_confidence_low": False,
        },
    ]

    legs_df = pd.DataFrame(legs_raw)
    return legs_df, historical


def run_pipeline(
    contest_type: str = "power_2_pick",
    config: Optional[PipelineConfig] = None,
    reference_now: Optional[datetime] = None,
) -> None:
    """
    Execute the full v1.5 quant pick'em pipeline using mock data.

    Prints segment metrics, blacklist status, chosen pair (or NO BET), EV,
    and Kelly size.
    """
    config = config or PipelineConfig()
    now = reference_now or datetime(2026, 2, 8, 12, 0, 0)

    print("=" * 70)
    print("  v1.5 Quant Pick'em Pipeline")
    print(f"  Contest type : {contest_type}")
    print(f"  Payout mult  : {payout_multiplier(contest_type, config):.2f}x")
    print(f"  Reference now: {now.isoformat()}")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # 1. Build mock data
    # ------------------------------------------------------------------ #
    legs_df, historical = _build_mock_data(config, now)

    # ------------------------------------------------------------------ #
    # 2. Fit segmented calibrators
    # ------------------------------------------------------------------ #
    seg_cal = SegmentedCalibrator(config)
    for seg_key, (preds, outcomes, retrained) in historical.items():
        seg_cal.fit_segment(seg_key, preds, outcomes, retrained_date=retrained)

    audit_df = seg_cal.audit()
    print("\n--- Segment Calibration Audit ---")
    print(audit_df.to_string(index=False))

    print("\n--- Blacklisted Segments ---")
    if seg_cal.blacklisted:
        for s in sorted(seg_cal.blacklisted):
            ece_val = seg_cal._segment_ece.get(s, float("nan"))
            print(f"  {s:20s}  ECE={ece_val:.4f}  (threshold={config.ece_blacklist_threshold})")
    else:
        print("  (none)")

    # ------------------------------------------------------------------ #
    # 3. Process each leg
    # ------------------------------------------------------------------ #
    processed_rows = []
    for _, row in legs_df.iterrows():
        sport = row["sport"]
        prop_type = row["prop_type"]

        calibrator, resolved_seg, is_stale = seg_cal.resolve(
            sport, prop_type, now=now,
        )

        p_shrunk = process_leg_probability(
            raw_prob=row["raw_prob"],
            sample_size=row["sample_size"],
            news_confidence_low=row["news_confidence_low"],
            calibrator=calibrator,
            seg_calibrator=seg_cal,
            config=config,
        )

        # Edge = p_shrunk - implied_fair (assume implied fair = 1/mult for demo)
        mult = payout_multiplier(contest_type, config)
        implied_fair = 1.0 / mult
        edge = p_shrunk - implied_fair

        is_blacklisted = resolved_seg in seg_cal.blacklisted

        quality_flag = (
            not is_stale
            and not is_blacklisted
            and calibrator is not None
        )

        processed_rows.append({
            "leg_id": row["leg_id"],
            "game_id": row["game_id"],
            "player_id": row["player_id"],
            "sport": sport,
            "prop_type": prop_type,
            "raw_prob": row["raw_prob"],
            "p_shrunk": p_shrunk,
            "segment_key": resolved_seg,
            "stale": is_stale,
            "blacklisted": is_blacklisted,
            "quality_flag": quality_flag,
            "edge": edge,
        })

    legs_processed = pd.DataFrame(processed_rows)

    print("\n--- Processed Legs ---")
    display_cols = [
        "leg_id", "sport", "prop_type", "raw_prob", "p_shrunk",
        "segment_key", "stale", "blacklisted", "quality_flag", "edge",
    ]
    print(legs_processed[display_cols].to_string(index=False))

    # ------------------------------------------------------------------ #
    # 4. Portfolio optimization
    # ------------------------------------------------------------------ #
    optimizer = PortfolioOptimizer(config)
    best = optimizer.optimize(legs_processed, contest_type, seg_cal)

    print("\n--- Portfolio Decision ---")

    # No-bet checks
    valid_count = legs_processed[
        (legs_processed["edge"] > 0)
        & (legs_processed["quality_flag"] == True)  # noqa: E712
        & (~legs_processed["segment_key"].isin(seg_cal.blacklisted))
        & (legs_processed["stale"] == False)  # noqa: E712
    ].shape[0]

    if valid_count < 2:
        print("  Result: NO BET  (fewer than 2 valid legs)")
        return

    if best is None or best["ev"] <= 0:
        print("  Result: NO BET  (no positive-EV pair found)")
        return

    # Check stale/blacklisted for best pair legs
    for tag in ("leg1", "leg2"):
        seg = best[tag]["segment_key"]
        if seg in seg_cal.blacklisted or best[tag].get("stale", False):
            print(f"  Result: NO BET  ({tag} depends on stale/blacklisted segment '{seg}')")
            return

    l1 = best["leg1"]
    l2 = best["leg2"]
    print(f"  Leg 1     : {l1['leg_id']}  ({l1['sport']}_{l1['prop_type']})  p={l1['p_shrunk']:.4f}")
    print(f"  Leg 2     : {l2['leg_id']}  ({l2['sport']}_{l2['prop_type']})  p={l2['p_shrunk']:.4f}")
    print(f"  Joint Prob: {best['joint_prob']:.4f}")
    print(f"  Payout    : {best['payout_multiplier']:.2f}x  ({best['contest_type']})")
    print(f"  EV        : {best['ev']:+.4f}")
    print(f"  Kelly f*  : {best['kelly_fraction']:.4f}  ({best['kelly_fraction']*100:.2f}% of bankroll)")
    print("=" * 70)


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    run_pipeline()
