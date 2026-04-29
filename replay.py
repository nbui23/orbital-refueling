"""CSV replay loading and drift summaries for synthetic telemetry."""
from __future__ import annotations

import pandas as pd

from detector import FEATURE_COLS


REQUIRED_COLUMNS = ["phase", "time", *FEATURE_COLS, "total_mass_transferred", "system_mode"]


def validate_schema(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Replay CSV missing required columns: {missing}")
    numeric_cols = [col for col in REQUIRED_COLUMNS if col not in ("phase", "system_mode")]
    non_numeric = [col for col in numeric_cols if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        raise ValueError(f"Replay CSV columns must be numeric: {non_numeric}")
    if df["time"].isna().any() or not df["time"].is_monotonic_increasing:
        raise ValueError("Replay CSV time column must be non-null and monotonically increasing.")


def load_replay_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_schema(df)
    return df[REQUIRED_COLUMNS].copy()


def drift_summary(replay_df: pd.DataFrame, nominal_df: pd.DataFrame) -> pd.DataFrame:
    validate_schema(replay_df)
    validate_schema(nominal_df)
    rows: list[dict[str, float | str]] = []
    for feature in FEATURE_COLS:
        nominal_mean = float(nominal_df[feature].mean())
        nominal_std = float(nominal_df[feature].std() or 1e-6)
        replay_mean = float(replay_df[feature].mean())
        rows.append(
            {
                "signal": feature,
                "nominal_mean": round(nominal_mean, 4),
                "replay_mean": round(replay_mean, 4),
                "mean_shift": round(replay_mean - nominal_mean, 4),
                "z_shift": round((replay_mean - nominal_mean) / nominal_std, 4),
            }
        )
    return pd.DataFrame(rows).sort_values("z_shift", key=lambda s: s.abs(), ascending=False)
