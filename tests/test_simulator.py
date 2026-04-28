"""Tests for simulator.py"""
import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator import (
    ANOMALY_SCENARIOS,
    PHASE_DURATIONS,
    PHASES,
    generate_telemetry,
)

EXPECTED_COLS = [
    "phase", "time", "arm_joint_angle", "arm_joint_velocity", "arm_motor_current",
    "end_effector_position_error", "interface_force", "interface_torque",
    "seal_pressure", "donor_tank_pressure", "receiver_tank_pressure",
    "line_pressure", "flow_rate", "total_mass_transferred", "pump_current",
    "propellant_temperature", "line_temperature", "attitude_error",
    "reaction_wheel_speed", "bus_voltage", "system_mode",
]

EXPECTED_ROWS = int(sum(PHASE_DURATIONS.values()) * 2)  # 2 Hz


class TestGenerateTelemetry:
    def test_returns_dataframe(self):
        df = generate_telemetry("nominal")
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self):
        df = generate_telemetry("nominal")
        assert len(df) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, got {len(df)}"

    def test_columns_present(self):
        df = generate_telemetry("nominal")
        for col in EXPECTED_COLS:
            assert col in df.columns, f"Missing column: {col}"

    def test_all_phases_present(self):
        df = generate_telemetry("nominal")
        for phase in PHASES:
            assert phase in df["phase"].values, f"Phase '{phase}' not in DataFrame"

    def test_phase_order(self):
        df = generate_telemetry("nominal")
        seen = df["phase"].unique().tolist()
        # Phases should appear in the canonical order
        idx = [PHASES.index(p) for p in seen]
        assert idx == sorted(idx), "Phases not in expected order"

    def test_time_monotonic(self):
        df = generate_telemetry("nominal")
        assert (df["time"].diff().dropna() > 0).all(), "Time not monotonically increasing"

    def test_no_nan(self):
        df = generate_telemetry("nominal")
        numeric = df.select_dtypes(include="number")
        assert not numeric.isnull().any().any(), "NaN values found in numeric columns"

    def test_mass_transferred_nonnegative(self):
        df = generate_telemetry("nominal")
        assert (df["total_mass_transferred"] >= 0).all()

    def test_mass_transferred_increases_during_transfer(self):
        df = generate_telemetry("nominal")
        transfer = df[df["phase"] == "main_transfer"]["total_mass_transferred"]
        assert transfer.iloc[-1] > transfer.iloc[0], "Mass should increase during main_transfer"

    def test_bus_voltage_reasonable(self):
        df = generate_telemetry("nominal")
        assert df["bus_voltage"].between(26, 30).all(), "Bus voltage out of expected range"

    def test_all_scenarios_run(self):
        for scenario in ANOMALY_SCENARIOS:
            df = generate_telemetry(scenario)
            assert len(df) == EXPECTED_ROWS, f"Wrong row count for scenario '{scenario}'"

    def test_invalid_scenario_raises(self):
        with pytest.raises(ValueError):
            generate_telemetry("unknown_anomaly_xyz")

    def test_reproducible_with_same_seed(self):
        df1 = generate_telemetry("nominal", seed=99)
        df2 = generate_telemetry("nominal", seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = generate_telemetry("nominal", seed=1)
        df2 = generate_telemetry("nominal", seed=2)
        assert not df1["attitude_error"].equals(df2["attitude_error"])


class TestAnomalyInjection:
    def _compare(self, signal: str, anomaly: str, phase: str):
        nom = generate_telemetry("nominal", seed=42)
        anom = generate_telemetry(anomaly, seed=42)
        nom_val = nom[nom["phase"] == phase][signal].mean()
        anom_val = anom[anom["phase"] == phase][signal].mean()
        return nom_val, anom_val

    def test_slow_leak_reduces_seal_pressure(self):
        nom, anom = self._compare("seal_pressure", "slow_leak", "main_transfer")
        assert anom < nom, "slow_leak should reduce seal_pressure"

    def test_partial_blockage_reduces_flow(self):
        nom, anom = self._compare("flow_rate", "partial_blockage", "main_transfer")
        assert anom < nom * 0.8, "partial_blockage should reduce flow_rate significantly"

    def test_partial_blockage_raises_pump_current(self):
        nom, anom = self._compare("pump_current", "partial_blockage", "main_transfer")
        assert anom > nom, "partial_blockage should raise pump_current"

    def test_arm_misalignment_keeps_position_error_high(self):
        nom, anom = self._compare(
            "end_effector_position_error", "arm_misalignment", "docking"
        )
        assert anom > nom * 5, "arm_misalignment should keep position_error high"

    def test_sensor_drift_raises_line_pressure(self):
        nom, anom = self._compare("line_pressure", "sensor_drift", "main_transfer")
        assert anom > nom, "sensor_drift should raise line_pressure reading"

    def test_unstable_slosh_increases_flow_variance(self):
        nom_df = generate_telemetry("nominal", seed=42)
        slosh_df = generate_telemetry("unstable_slosh", seed=42)
        nom_std = nom_df[nom_df["phase"] == "main_transfer"]["flow_rate"].std()
        slosh_std = slosh_df[slosh_df["phase"] == "main_transfer"]["flow_rate"].std()
        assert slosh_std > nom_std * 2, "unstable_slosh should increase flow_rate variance"
