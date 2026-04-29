#!/usr/bin/env python3
"""Run deterministic validation summaries for all Orbital Refueling Telemetry Simulator scenarios."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector import EnsemblePhaseAwareDetector, PhaseAwareDetector
from estimator import estimate_line_pressure
from explainer import explain_window
from rules import evaluate_rules, group_alerts
from simulator import ANOMALY_SCENARIOS, generate_telemetry
from sequence_detector import RollingWindowAnomalyDetector

OUTPUT_DIR = ROOT / "outputs"
DETECTOR_PATH = OUTPUT_DIR / "default_phase_aware_detector.pkl"
CSV_PATH = OUTPUT_DIR / "scenario_validation.csv"

TRAINING_SEEDS = range(8)
SCENARIO_SEED = 42
ML_THRESHOLD = 0.45
EXPLAIN_MAX_SAMPLES = 16
SEVERITY_ORDER = {"NONE": 0, "WARNING": 1, "CRITICAL": 2}


def _train_default_detector() -> PhaseAwareDetector:
    nominal_runs = [
        generate_telemetry("nominal", seed=seed)
        for seed in TRAINING_SEEDS
    ]
    training_df = pd.concat(nominal_runs, ignore_index=True)
    return PhaseAwareDetector(contamination=0.02, n_estimators=200).fit(training_df)


def load_or_train_detector() -> PhaseAwareDetector:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if DETECTOR_PATH.exists():
        with DETECTOR_PATH.open("rb") as f:
            return pickle.load(f)

    detector = _train_default_detector()
    with DETECTOR_PATH.open("wb") as f:
        pickle.dump(detector, f)
    return detector


def _highest_severity(severities: Iterable[str]) -> str:
    highest = "NONE"
    for severity in severities:
        if SEVERITY_ORDER.get(severity, 0) > SEVERITY_ORDER[highest]:
            highest = severity
    return highest


def _top_contributing_signal(contribs: dict[str, float]) -> str:
    if not contribs:
        return ""
    return max(contribs.items(), key=lambda item: item[1])[0]


def evaluate_scenario(detector: PhaseAwareDetector, scenario: str) -> dict[str, object]:
    df = generate_telemetry(scenario, seed=SCENARIO_SEED)
    scores = detector.score(df)
    score_summary = evaluate_scenario.ensemble.score_summary(df)  # type: ignore[attr-defined]
    sequence_scores = evaluate_scenario.sequence_detector.score(df)  # type: ignore[attr-defined]
    estimates = estimate_line_pressure(df)
    raw_alerts = evaluate_rules(df)
    grouped_alerts = group_alerts(raw_alerts, gap_s=15.0)
    contribs = explain_window(
        detector,
        df,
        scores,
        threshold=ML_THRESHOLD,
        n_top=8,
        max_samples=EXPLAIN_MAX_SAMPLES,
    )

    return {
        "scenario": scenario,
        "max_ml_score": round(float(scores.max()), 3),
        "mean_ml_score": round(float(scores.mean()), 3),
        "mean_ml_uncertainty": round(float(score_summary["ml_score_uncertainty"].mean()), 3),
        "max_ml_uncertainty": round(float(score_summary["ml_score_uncertainty"].max()), 3),
        "max_sequence_score": round(float(sequence_scores.max()), 3),
        "mean_sequence_score": round(float(sequence_scores.mean()), 3),
        "max_pressure_residual": round(float(estimates["line_pressure_residual"].abs().max()), 3),
        "ml_anomaly_rate_pct": round(float((scores > ML_THRESHOLD).mean() * 100.0), 1),
        "raw_rule_alert_count": len(raw_alerts),
        "grouped_rule_alert_count": len(grouped_alerts),
        "highest_rule_severity": _highest_severity(alert.severity for alert in raw_alerts),
        "top_contributing_signal": _top_contributing_signal(contribs),
    }


def _print_table(results: pd.DataFrame) -> None:
    display = results.rename(columns={
        "scenario": "Scenario",
        "max_ml_score": "Max ML",
        "mean_ml_score": "Mean ML",
        "mean_ml_uncertainty": "Mean ML Unc",
        "max_ml_uncertainty": "Max ML Unc",
        "max_sequence_score": "Max Seq",
        "mean_sequence_score": "Mean Seq",
        "max_pressure_residual": "Max Pressure Residual",
        "ml_anomaly_rate_pct": "ML Rate %",
        "raw_rule_alert_count": "Raw Rules",
        "grouped_rule_alert_count": "Grouped Rules",
        "highest_rule_severity": "Highest Severity",
        "top_contributing_signal": "Top Signal",
    })
    print(display.to_string(index=False))


def main() -> None:
    detector = load_or_train_detector()
    evaluate_scenario.ensemble = EnsemblePhaseAwareDetector(seeds=(0, 1, 2, 3, 4), n_estimators=80).fit()  # type: ignore[attr-defined]
    nominal_runs = [generate_telemetry("nominal", seed=seed) for seed in TRAINING_SEEDS]
    evaluate_scenario.sequence_detector = RollingWindowAnomalyDetector().fit(  # type: ignore[attr-defined]
        pd.concat(nominal_runs, ignore_index=True)
    )
    results = pd.DataFrame(
        evaluate_scenario(detector, scenario)
        for scenario in ANOMALY_SCENARIOS
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(CSV_PATH, index=False)

    print(f"Orbital Refueling Telemetry Simulator scenario validation (ML threshold: {ML_THRESHOLD:.2f})")
    print(f"Detector: {DETECTOR_PATH.relative_to(ROOT)}")
    print(f"CSV: {CSV_PATH.relative_to(ROOT)}")
    print()
    _print_table(results)


if __name__ == "__main__":
    main()
