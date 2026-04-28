"""
detector.py
Phase-aware anomaly detector using IsolationForest.

One model is trained per operation phase on nominal telemetry.
At inference time, each row is scored against its phase-specific model.
This approach captures the fact that "normal" looks very different across phases
(e.g., zero flow during approach vs. 0.5 kg/s flow during main_transfer).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Numeric features fed to the model. Excludes phase/time/system_mode (categorical/index).
FEATURE_COLS: list[str] = [
    "arm_joint_angle",
    "arm_joint_velocity",
    "arm_motor_current",
    "end_effector_position_error",
    "interface_force",
    "interface_torque",
    "seal_pressure",
    "donor_tank_pressure",
    "receiver_tank_pressure",
    "line_pressure",
    "flow_rate",
    "pump_current",
    "propellant_temperature",
    "line_temperature",
    "attitude_error",
    "reaction_wheel_speed",
    "bus_voltage",
]

def _normalize(decision: np.ndarray, k: float = 20.0) -> np.ndarray:
    """Map IsolationForest decision_function values to [0, 1] anomaly probability.

    decision_function returns threshold-relative scores:
      > 0  → inlier (nominal)  → we want score near 0
      < 0  → outlier (anomaly) → we want score near 1
      = 0  → on the decision boundary → score = 0.5

    Sigmoid with negated input gives a smooth, self-calibrating mapping
    that does not depend on dataset-specific raw score ranges.
    """
    return 1.0 / (1.0 + np.exp(k * decision))


class PhaseAwareDetector:
    """
    Trains and applies one IsolationForest per operation phase.

    Usage
    -----
    detector = PhaseAwareDetector().fit(nominal_df)
    scores   = detector.score(test_df)  # np.ndarray, shape (n,), values in [0, 1]
    """

    def __init__(
        self,
        contamination: float = 0.02,
        n_estimators: int = 150,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self._models: dict[str, IsolationForest] = {}
        self._scalers: dict[str, StandardScaler] = {}

    def fit(self, nominal_df: pd.DataFrame) -> "PhaseAwareDetector":
        """Train one model per phase on nominal telemetry. Returns self."""
        for phase in nominal_df["phase"].unique():
            subset = nominal_df.loc[nominal_df["phase"] == phase, FEATURE_COLS].dropna()
            if len(subset) < 10:
                continue
            scaler = StandardScaler()
            X = scaler.fit_transform(subset.values)
            model = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X)
            self._models[phase] = model
            self._scalers[phase] = scaler
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Return per-row anomaly scores in [0, 1]. Higher = more anomalous.

        Rows whose phase has no trained model receive a score of 0.
        """
        scores = np.zeros(len(df), dtype=float)
        for phase in df["phase"].unique():
            if phase not in self._models:
                continue
            mask = (df["phase"] == phase).values
            X = df.loc[mask, FEATURE_COLS].fillna(0.0).values
            X_scaled = self._scalers[phase].transform(X)
            decision = self._models[phase].decision_function(X_scaled)
            scores[mask] = _normalize(decision)
        return scores

    def phase_means(self, phase: str) -> np.ndarray | None:
        """Return nominal feature means (original scale) for a phase, or None."""
        scaler = self._scalers.get(phase)
        return scaler.mean_ if scaler is not None else None

    def score_single(self, df_row: pd.DataFrame) -> float:
        """Score a single-row DataFrame. Convenience wrapper around score()."""
        return float(self.score(df_row)[0])

    @property
    def trained_phases(self) -> list[str]:
        return list(self._models.keys())
