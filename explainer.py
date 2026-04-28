"""
explainer.py
Perturbation-based feature contribution analysis for anomalous telemetry.

Method: for each feature in an anomalous row, replace it with the nominal
phase mean (learned by the scaler during training) and measure how much
the anomaly score drops. A large drop indicates that feature is driving
the anomaly — analogous to a SHAP baseline explanation without the library.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from detector import FEATURE_COLS, PhaseAwareDetector


def explain_row(
    detector: PhaseAwareDetector,
    df: pd.DataFrame,
    idx: int,
    n_top: int = 6,
) -> dict[str, float]:
    """
    Compute per-feature anomaly contributions for a single row.

    Returns
    -------
    dict mapping feature name → contribution score (descending order).
    Positive value: feature is pushing the anomaly score up.
    Zero or negative: feature looks nominal for its phase.
    """
    row = df.iloc[[idx]]
    phase = str(row["phase"].values[0])

    if phase not in detector.trained_phases:
        return {}

    baseline = detector.score_single(row)
    nominal_means = detector.phase_means(phase)
    if nominal_means is None:
        return {}

    features = row[FEATURE_COLS].fillna(0.0).values[0].astype(float)
    contributions: dict[str, float] = {}

    for i, feat in enumerate(FEATURE_COLS):
        modified = features.copy()
        modified[i] = nominal_means[i]
        mod_df = row[FEATURE_COLS].copy()
        mod_df.iloc[0, i] = nominal_means[i]
        # Rebuild a minimal DataFrame with required columns
        tmp = row.copy()
        tmp[FEATURE_COLS] = mod_df.values
        replaced_score = detector.score_single(tmp)
        contributions[feat] = float(baseline - replaced_score)

    top = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)[:n_top]
    return dict(top)


def explain_window(
    detector: PhaseAwareDetector,
    df: pd.DataFrame,
    scores: np.ndarray,
    threshold: float = 0.45,
    n_top: int = 6,
    max_samples: int = 40,
) -> dict[str, float]:
    """
    Aggregate feature contributions over rows where anomaly score exceeds threshold.

    Averages contributions across sampled high-score rows for a summary view.
    Returns empty dict if no anomalous rows found.
    """
    high_idx = np.where(scores > threshold)[0]
    if len(high_idx) == 0:
        return {}

    # Sample to bound runtime
    if len(high_idx) > max_samples:
        rng = np.random.default_rng(42)
        high_idx = rng.choice(high_idx, size=max_samples, replace=False)

    agg: dict[str, list[float]] = {f: [] for f in FEATURE_COLS}
    for idx in high_idx:
        contribs = explain_row(detector, df, int(idx), n_top=len(FEATURE_COLS))
        for feat, val in contribs.items():
            agg[feat].append(val)

    mean_contribs = {
        feat: float(np.mean(vals))
        for feat, vals in agg.items()
        if vals
    }
    top = sorted(mean_contribs.items(), key=lambda kv: kv[1], reverse=True)[:n_top]
    return dict(top)
