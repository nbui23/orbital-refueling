"""Tests for estimator, temporal detector, and replay ingestion."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from estimator import estimate_line_pressure
from replay import drift_summary, load_replay_csv, validate_schema
from sequence_detector import RollingWindowAnomalyDetector
from simulator import generate_telemetry


def test_estimator_outputs_residuals_and_tracks_length():
    df = generate_telemetry("nominal", seed=0)
    estimates = estimate_line_pressure(df)
    assert len(estimates) == len(df)
    assert "line_pressure_residual" in estimates.columns
    assert estimates["line_pressure_residual"].abs().mean() < 5.0


def test_estimator_residual_larger_for_dropout():
    nominal = estimate_line_pressure(generate_telemetry("nominal", seed=42))
    dropout = estimate_line_pressure(generate_telemetry("sensor_dropout", seed=42))
    nominal_transfer = nominal[(nominal["time"] >= 225) & (nominal["time"] <= 255)]
    dropout_transfer = dropout[(dropout["time"] >= 225) & (dropout["time"] <= 255)]
    assert dropout_transfer["line_pressure_residual"].abs().max() > nominal_transfer["line_pressure_residual"].abs().max() * 3


def test_sequence_detector_scores_temporal_fault_above_nominal():
    train = pd.concat([generate_telemetry("nominal", seed=i) for i in range(3)], ignore_index=True)
    detector = RollingWindowAnomalyDetector().fit(train)
    nominal_scores = detector.score(generate_telemetry("nominal", seed=42))
    bias_scores = detector.score(generate_telemetry("bias_oscillation", seed=42))
    assert bias_scores.mean() > nominal_scores.mean()


def test_replay_loads_example_csv():
    df = load_replay_csv("examples/replay_sample.csv")
    assert len(df) == 3
    validate_schema(df)


def test_replay_schema_validation_reports_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        validate_schema(pd.DataFrame({"phase": ["approach"], "time": [0.0]}))


def test_drift_summary_sorts_shifted_signals():
    nominal = generate_telemetry("nominal", seed=0)
    replay = nominal.copy()
    replay["line_pressure"] = replay["line_pressure"] + 20.0
    summary = drift_summary(replay, nominal)
    assert summary.iloc[0]["signal"] == "line_pressure"
