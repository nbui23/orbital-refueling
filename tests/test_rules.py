"""Tests for rules.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest

from rules import RULES, RuleAlert, alerts_to_dataframe, evaluate_rules


def _make_row(**kwargs) -> pd.DataFrame:
    defaults = {
        "phase": "main_transfer",
        "time": 100.0,
        "attitude_error": 0.3,
        "bus_voltage": 28.0,
        "seal_pressure": 5.0,
        "flow_rate": 0.0,
        "line_pressure": 150.0,
        "pump_current": 8.0,
        "interface_force": 50.0,
        "end_effector_position_error": 0.2,
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


class TestRuleDefinitions:
    def test_all_rules_have_required_fields(self):
        for rule in RULES:
            assert rule.name
            assert rule.description
            assert rule.signal
            assert rule.severity in ("WARNING", "CRITICAL")
            assert callable(rule.condition)

    def test_no_duplicate_rule_names(self):
        names = [r.name for r in RULES]
        assert len(names) == len(set(names)), "Duplicate rule names found"


class TestEvaluateRules:
    def test_no_alerts_nominal(self):
        df = _make_row()
        alerts = evaluate_rules(df)
        assert alerts == []

    def test_attitude_critical_fires(self):
        df = _make_row(attitude_error=6.0, phase="main_transfer")
        alerts = evaluate_rules(df)
        names = [a.rule_name for a in alerts]
        assert "ATTITUDE_CRITICAL" in names

    def test_attitude_warning_fires(self):
        df = _make_row(attitude_error=3.0, phase="approach")
        alerts = evaluate_rules(df)
        names = [a.rule_name for a in alerts]
        assert "ATTITUDE_WARNING" in names

    def test_attitude_critical_not_warning_when_critical(self):
        # Both should fire (critical > warning threshold)
        df = _make_row(attitude_error=6.0)
        alerts = evaluate_rules(df)
        names = [a.rule_name for a in alerts]
        assert "ATTITUDE_CRITICAL" in names

    def test_seal_pressure_only_active_in_correct_phases(self):
        # Should NOT fire in approach
        df_approach = _make_row(phase="approach", seal_pressure=1.0)
        alerts = evaluate_rules(df_approach)
        names = [a.rule_name for a in alerts]
        assert "SEAL_PRESSURE_CRITICAL" not in names

        # Should fire in main_transfer
        df_transfer = _make_row(phase="main_transfer", seal_pressure=1.0)
        alerts = evaluate_rules(df_transfer)
        names = [a.rule_name for a in alerts]
        assert "SEAL_PRESSURE_CRITICAL" in names

    def test_leak_detected_fires_in_leak_check(self):
        df = _make_row(phase="leak_check", flow_rate=0.10)
        alerts = evaluate_rules(df)
        names = [a.rule_name for a in alerts]
        assert "LEAK_DETECTED" in names

    def test_leak_detected_not_in_transfer(self):
        df = _make_row(phase="main_transfer", flow_rate=0.5)
        alerts = evaluate_rules(df)
        names = [a.rule_name for a in alerts]
        assert "LEAK_DETECTED" not in names

    def test_pump_overcurrent_critical(self):
        df = _make_row(phase="main_transfer", pump_current=16.0)
        alerts = evaluate_rules(df)
        names = [a.rule_name for a in alerts]
        assert "PUMP_OVERCURRENT_CRITICAL" in names

    def test_line_pressure_critical(self):
        df = _make_row(phase="main_transfer", line_pressure=210.0)
        alerts = evaluate_rules(df)
        names = [a.rule_name for a in alerts]
        assert "LINE_PRESSURE_CRITICAL" in names

    def test_line_pressure_inactive_in_approach(self):
        df = _make_row(phase="approach", line_pressure=999.0)
        alerts = evaluate_rules(df)
        names = [a.rule_name for a in alerts]
        assert "LINE_PRESSURE_CRITICAL" not in names

    def test_bus_voltage_critical(self):
        df = _make_row(bus_voltage=25.0)
        alerts = evaluate_rules(df)
        names = [a.rule_name for a in alerts]
        assert "BUS_VOLTAGE_CRITICAL" in names

    def test_interface_force_critical_in_docking(self):
        df = _make_row(phase="docking", interface_force=95.0)
        alerts = evaluate_rules(df)
        names = [a.rule_name for a in alerts]
        assert "INTERFACE_FORCE_CRITICAL" in names

    def test_arm_position_critical_in_docking(self):
        df = _make_row(phase="docking", end_effector_position_error=30.0)
        alerts = evaluate_rules(df)
        names = [a.rule_name for a in alerts]
        assert "ARM_POSITION_CRITICAL" in names

    def test_deduplication_across_timesteps(self):
        # Same violation at two consecutive timesteps → only one alert
        row1 = _make_row(time=1.0, attitude_error=6.0)
        row2 = _make_row(time=1.5, attitude_error=6.0)
        df = pd.concat([row1, row2], ignore_index=True)
        alerts = evaluate_rules(df)
        critical = [a for a in alerts if a.rule_name == "ATTITUDE_CRITICAL"]
        assert len(critical) == 1, "Duplicate alert for continuous violation"

    def test_alert_has_correct_fields(self):
        df = _make_row(attitude_error=6.0, phase="approach", time=5.0)
        alerts = evaluate_rules(df)
        a = next(a for a in alerts if a.rule_name == "ATTITUDE_CRITICAL")
        assert a.severity == "CRITICAL"
        assert a.signal == "attitude_error"
        assert a.time == 5.0
        assert a.phase == "approach"


class TestAlertsToDataframe:
    def test_empty_list_returns_empty_df(self):
        df = alerts_to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_converts_alerts(self):
        alert = RuleAlert(
            rule_name="TEST", description="desc", severity="WARNING",
            signal="flow_rate", value=0.1, phase="leak_check", time=200.0,
        )
        df = alerts_to_dataframe([alert])
        assert len(df) == 1
        assert df.iloc[0]["severity"] == "WARNING"
