"""Lightweight state estimation helpers for synthetic telemetry."""
from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_line_pressure(
    df: pd.DataFrame,
    process_var: float = 0.08,
    measurement_var: float = 6.0,
) -> pd.DataFrame:
    """Estimate line pressure with a simple one-dimensional Kalman filter.

    The model assumes pressure changes slowly except at phase transitions. It is
    intentionally small and transparent for this prototype.
    """
    observed = df["line_pressure"].astype(float).to_numpy()
    phases = df["phase"].astype(str).to_numpy()
    estimate = np.zeros(len(df), dtype=float)
    residual = np.zeros(len(df), dtype=float)

    x = observed[0]
    p = 1.0
    last_phase = phases[0]

    for i, z in enumerate(observed):
        q = process_var * (25.0 if phases[i] != last_phase else 1.0)
        p = p + q
        k = p / (p + measurement_var)
        x = x + k * (z - x)
        p = (1.0 - k) * p
        estimate[i] = x
        residual[i] = z - x
        last_phase = phases[i]

    return pd.DataFrame(
        {
            "time": df["time"].values,
            "phase": df["phase"].values,
            "line_pressure_observed": observed,
            "line_pressure_estimated": estimate,
            "line_pressure_residual": residual,
        },
        index=df.index,
    )
