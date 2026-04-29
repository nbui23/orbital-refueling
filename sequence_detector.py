"""Experimental sequence-aware anomaly scoring via rolling-window features."""
from __future__ import annotations

import numpy as np
import pandas as pd


TEMPORAL_FEATURES = ["line_pressure", "flow_rate", "pump_current"]


class RollingWindowAnomalyDetector:
    """Temporal baseline using rolling mean/std features calibrated on nominal data."""

    def __init__(self, window: int = 12, features: list[str] | None = None) -> None:
        self.window = window
        self.features = features or TEMPORAL_FEATURES
        self._means: dict[str, pd.Series] = {}
        self._stds: dict[str, pd.Series] = {}

    def _temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for feature in self.features:
            series = df[feature].astype(float)
            rolling = series.rolling(self.window, min_periods=2)
            out[f"{feature}_roll_mean"] = rolling.mean().fillna(series)
            out[f"{feature}_roll_std"] = rolling.std().fillna(0.0)
            out[f"{feature}_delta"] = series.diff().fillna(0.0)
        return out

    def fit(self, nominal_df: pd.DataFrame) -> "RollingWindowAnomalyDetector":
        feats = self._temporal_features(nominal_df)
        for phase in nominal_df["phase"].unique():
            subset = feats.loc[nominal_df["phase"] == phase]
            self._means[phase] = subset.mean()
            self._stds[phase] = subset.std().replace(0.0, 1e-6).fillna(1e-6)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        feats = self._temporal_features(df)
        scores = np.zeros(len(df), dtype=float)
        for phase in df["phase"].unique():
            if phase not in self._means:
                continue
            mask = (df["phase"] == phase).values
            z = (feats.loc[mask] - self._means[phase]) / self._stds[phase]
            distance = np.sqrt((z**2).mean(axis=1)).to_numpy()
            scores[mask] = 1.0 - np.exp(-distance / 3.0)
        return np.clip(scores, 0.0, 1.0)
