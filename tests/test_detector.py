"""Tests for detector.py and explainer.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from detector import FEATURE_COLS, PhaseAwareDetector
from explainer import explain_row, explain_window
from simulator import PHASES, generate_telemetry


@pytest.fixture(scope="module")
def nominal_df():
    return generate_telemetry("nominal", seed=0)


@pytest.fixture(scope="module")
def trained_detector(nominal_df):
    return PhaseAwareDetector(contamination=0.02, n_estimators=50).fit(nominal_df)


@pytest.fixture(scope="module")
def anomaly_df():
    return generate_telemetry("pump_degradation", seed=42)


class TestPhaseAwareDetector:
    def test_fit_trains_all_phases(self, trained_detector):
        for phase in PHASES:
            assert phase in trained_detector.trained_phases, f"No model for phase '{phase}'"

    def test_score_returns_array(self, trained_detector, nominal_df):
        scores = trained_detector.score(nominal_df)
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (len(nominal_df),)

    def test_scores_in_range(self, trained_detector, nominal_df):
        scores = trained_detector.score(nominal_df)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_nominal_scores_low(self, trained_detector, nominal_df):
        scores = trained_detector.score(nominal_df)
        # Nominal data should have low average anomaly score
        assert scores.mean() < 0.35, f"Nominal mean score too high: {scores.mean():.3f}"

    def test_anomaly_scores_higher_than_nominal(self, trained_detector, nominal_df, anomaly_df):
        nom_scores = trained_detector.score(nominal_df)
        anom_scores = trained_detector.score(anomaly_df)
        # Anomaly mean should exceed nominal mean
        assert anom_scores.mean() > nom_scores.mean(), (
            f"Anomaly mean {anom_scores.mean():.3f} not above nominal {nom_scores.mean():.3f}"
        )

    def test_score_single_matches_score(self, trained_detector, nominal_df):
        row = nominal_df.iloc[[5]]
        score_batch = trained_detector.score(row)[0]
        score_single = trained_detector.score_single(row)
        assert abs(score_batch - score_single) < 1e-9

    def test_unknown_phase_scores_zero(self, trained_detector, nominal_df):
        df_copy = nominal_df.iloc[[0]].copy()
        df_copy["phase"] = "unknown_phase_xyz"
        scores = trained_detector.score(df_copy)
        assert scores[0] == 0.0

    def test_phase_means_returns_array(self, trained_detector):
        means = trained_detector.phase_means("main_transfer")
        assert means is not None
        assert len(means) == len(FEATURE_COLS)

    def test_phase_means_unknown_phase_returns_none(self, trained_detector):
        assert trained_detector.phase_means("nonexistent") is None

    def test_feature_cols_all_present_in_nominal(self, nominal_df):
        for col in FEATURE_COLS:
            assert col in nominal_df.columns, f"FEATURE_COLS references missing column: {col}"

    def test_pump_degradation_peaks_in_main_transfer(self, trained_detector):
        anom_df = generate_telemetry("pump_degradation", seed=42)
        scores = trained_detector.score(anom_df)
        transfer_mask = anom_df["phase"] == "main_transfer"
        other_mask = ~transfer_mask & anom_df["phase"].isin(["approach", "arm_alignment"])
        # main_transfer should have higher mean score than quiet phases
        assert scores[transfer_mask].mean() > scores[other_mask].mean()


class TestExplainer:
    def test_explain_row_returns_dict(self, trained_detector, anomaly_df):
        result = explain_row(trained_detector, anomaly_df, 0)
        assert isinstance(result, dict)

    def test_explain_row_keys_are_feature_cols(self, trained_detector, anomaly_df):
        result = explain_row(trained_detector, anomaly_df, 0, n_top=5)
        for key in result:
            assert key in FEATURE_COLS

    def test_explain_row_length_bounded(self, trained_detector, anomaly_df):
        result = explain_row(trained_detector, anomaly_df, 0, n_top=4)
        assert len(result) <= 4

    def test_explain_row_unknown_phase_returns_empty(self, trained_detector, nominal_df):
        df_copy = nominal_df.iloc[[0]].copy()
        df_copy["phase"] = "ghost_phase"
        result = explain_row(trained_detector, df_copy, 0)
        assert result == {}

    def test_explain_window_returns_dict(self, trained_detector, anomaly_df):
        scores = trained_detector.score(anomaly_df)
        result = explain_window(trained_detector, anomaly_df, scores, threshold=0.3)
        assert isinstance(result, dict)

    def test_explain_window_empty_when_no_anomalies(self, trained_detector, nominal_df):
        # All scores at 0 → nothing above threshold
        scores = np.zeros(len(nominal_df))
        result = explain_window(trained_detector, nominal_df, scores, threshold=0.9)
        assert result == {}

    def test_pump_degradation_highlights_pump_signal(self, trained_detector):
        anom_df = generate_telemetry("pump_degradation", seed=42)
        scores = trained_detector.score(anom_df)
        contribs = explain_window(trained_detector, anom_df, scores, threshold=0.3)
        if contribs:
            top_features = list(contribs.keys())
            pump_related = {"pump_current", "flow_rate", "line_temperature"}
            assert any(f in pump_related for f in top_features), (
                f"Expected pump-related feature in top contributors, got: {top_features}"
            )
