"""
rules.py
Deterministic safety rule engine for orbital refueling operations.

Design principle: hard limits are non-negotiable. The ML detector supplements
these rules by catching subtle multivariate drift before thresholds are hit.
Rules never delegate to ML; ML never overrides rules.

Two alert representations are provided:
- evaluate_rules() → list[RuleAlert]   (timestamp-level, used by tests and ML overlay)
- group_alerts()   → pd.DataFrame      (event-window summary, used by the dashboard)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd


@dataclass(frozen=True)
class Rule:
    name: str
    description: str
    signal: str
    condition: Callable[[float], bool]
    severity: str  # "WARNING" | "CRITICAL"
    phases: Optional[tuple[str, ...]] = None  # None = active in all phases


@dataclass
class RuleAlert:
    rule_name: str
    description: str
    severity: str
    signal: str
    value: float
    phase: str
    time: float


RULES: list[Rule] = [
    # ── Attitude ──────────────────────────────────────────────────────────────
    Rule(
        name="ATTITUDE_CRITICAL",
        description="Attitude error > 5°. Structural alignment unsafe — abort operation.",
        signal="attitude_error",
        condition=lambda v: v > 5.0,
        severity="CRITICAL",
    ),
    Rule(
        name="ATTITUDE_WARNING",
        description="Attitude error > 2°. Monitor attitude control system.",
        signal="attitude_error",
        condition=lambda v: v > 2.0,
        severity="WARNING",
    ),
    # ── Bus Voltage ───────────────────────────────────────────────────────────
    Rule(
        name="BUS_VOLTAGE_CRITICAL",
        description="Bus voltage < 26 V. Power fault — abort all operations immediately.",
        signal="bus_voltage",
        condition=lambda v: v < 26.0,
        severity="CRITICAL",
    ),
    Rule(
        name="BUS_VOLTAGE_WARNING",
        description="Bus voltage < 27 V. Investigate power draw.",
        signal="bus_voltage",
        condition=lambda v: v < 27.0,
        severity="WARNING",
    ),
    # ── Seal ─────────────────────────────────────────────────────────────────
    Rule(
        name="SEAL_PRESSURE_CRITICAL",
        description="Seal pressure < 4 bar after seal established. Seal failure risk — stop transfer.",
        signal="seal_pressure",
        condition=lambda v: v < 4.0,
        severity="CRITICAL",
        # seal_check excluded: pressure intentionally ramps 0→5 bar during that phase.
        # A drop below 4 bar after seal_check signals a developing failure.
        phases=("pressure_equalization", "main_transfer", "leak_check"),
    ),
    # ── Leak Check ────────────────────────────────────────────────────────────
    Rule(
        name="LEAK_DETECTED",
        description="Flow > 0.04 kg/s during leak check. Line breach suspected.",
        signal="flow_rate",
        condition=lambda v: abs(v) > 0.04,
        severity="CRITICAL",
        phases=("leak_check",),
    ),
    # ── Line Pressure ─────────────────────────────────────────────────────────
    Rule(
        name="LINE_PRESSURE_CRITICAL",
        description="Line pressure > 200 bar. Over-pressure risk — abort transfer.",
        signal="line_pressure",
        condition=lambda v: v > 200.0,
        severity="CRITICAL",
        phases=("pressure_equalization", "main_transfer", "leak_check"),
    ),
    Rule(
        name="LINE_PRESSURE_WARNING",
        description="Line pressure > 180 bar. Approaching structural limit.",
        signal="line_pressure",
        condition=lambda v: v > 180.0,
        severity="WARNING",
        phases=("pressure_equalization", "main_transfer", "leak_check"),
    ),
    # ── Pump ─────────────────────────────────────────────────────────────────
    Rule(
        name="PUMP_OVERCURRENT_CRITICAL",
        description="Pump current > 15 A. Pump failure risk — halt transfer.",
        signal="pump_current",
        condition=lambda v: v > 15.0,
        severity="CRITICAL",
        phases=("pressure_equalization", "main_transfer"),
    ),
    Rule(
        name="PUMP_OVERCURRENT_WARNING",
        description="Pump current > 12 A. Possible degradation or partial blockage.",
        signal="pump_current",
        condition=lambda v: v > 12.0,
        severity="WARNING",
        phases=("pressure_equalization", "main_transfer"),
    ),
    # ── Interface Force ───────────────────────────────────────────────────────
    Rule(
        name="INTERFACE_FORCE_CRITICAL",
        description="Interface force > 90 N. Structural limit risk — check alignment.",
        signal="interface_force",
        condition=lambda v: v > 90.0,
        severity="CRITICAL",
        phases=("docking", "seal_check", "pressure_equalization", "main_transfer", "leak_check"),
    ),
    Rule(
        name="INTERFACE_FORCE_WARNING",
        description="Interface force > 70 N. Elevated mechanical stress.",
        signal="interface_force",
        condition=lambda v: v > 70.0,
        severity="WARNING",
        phases=("docking", "seal_check", "pressure_equalization", "main_transfer", "leak_check"),
    ),
    # ── End Effector Position ─────────────────────────────────────────────────
    Rule(
        name="ARM_POSITION_CRITICAL",
        description="End effector error > 25 mm during contact phase. Abort docking.",
        signal="end_effector_position_error",
        condition=lambda v: v > 25.0,
        severity="CRITICAL",
        phases=("docking", "seal_check"),
    ),
    Rule(
        name="ARM_POSITION_WARNING",
        description="End effector error > 10 mm during active engagement.",
        signal="end_effector_position_error",
        condition=lambda v: v > 10.0,
        severity="WARNING",
        phases=("docking", "seal_check", "pressure_equalization", "main_transfer", "leak_check"),
    ),
]


def evaluate_rules(df: pd.DataFrame) -> list[RuleAlert]:
    """
    Evaluate all deterministic rules against a telemetry DataFrame.

    Returns a list of RuleAlert for every timestep where a rule fires.
    Adjacent duplicate alerts (same rule, consecutive timesteps) are deduplicated
    to a single alert per contiguous violation window.
    """
    alerts: list[RuleAlert] = []
    # Track last-seen violation per rule to avoid alert floods
    _last: dict[str, float] = {}

    for _, row in df.iterrows():
        phase = str(row["phase"])
        t = float(row["time"])

        for rule in RULES:
            if rule.phases is not None and phase not in rule.phases:
                continue
            if rule.signal not in row or pd.isna(row[rule.signal]):
                continue

            value = float(row[rule.signal])
            if not rule.condition(value):
                _last.pop(rule.name, None)  # reset dedup
                continue

            # Emit at most one alert per continuous violation window
            if rule.name in _last:
                continue

            _last[rule.name] = t
            alerts.append(
                RuleAlert(
                    rule_name=rule.name,
                    description=rule.description,
                    severity=rule.severity,
                    signal=rule.signal,
                    value=round(value, 4),
                    phase=phase,
                    time=t,
                )
            )

    return alerts


def alerts_to_dataframe(alerts: list[RuleAlert]) -> pd.DataFrame:
    """Convert a list of RuleAlert objects to a display-ready DataFrame."""
    if not alerts:
        return pd.DataFrame(
            columns=["time", "phase", "severity", "rule", "signal", "value", "description"]
        )
    return pd.DataFrame(
        [
            {
                "time": a.time,
                "phase": a.phase,
                "severity": a.severity,
                "rule": a.rule_name,
                "signal": a.signal,
                "value": a.value,
                "description": a.description,
            }
            for a in alerts
        ]
    )


# ── Human-readable titles and recommended actions per rule ────────────────────

_RULE_TITLES: dict[str, str] = {
    "ATTITUDE_CRITICAL":        "Attitude Error — Critical",
    "ATTITUDE_WARNING":         "Attitude Error — Warning",
    "BUS_VOLTAGE_CRITICAL":     "Bus Voltage — Critical",
    "BUS_VOLTAGE_WARNING":      "Bus Voltage — Warning",
    "SEAL_PRESSURE_CRITICAL":   "Seal Pressure Loss — Critical",
    "LEAK_DETECTED":            "Leak Detected — Critical",
    "LINE_PRESSURE_CRITICAL":   "Line Over-Pressure — Critical",
    "LINE_PRESSURE_WARNING":    "Line Pressure Elevated — Warning",
    "PUMP_OVERCURRENT_CRITICAL":"Pump Overcurrent — Critical",
    "PUMP_OVERCURRENT_WARNING": "Pump Overcurrent — Warning",
    "INTERFACE_FORCE_CRITICAL": "Interface Force Overload — Critical",
    "INTERFACE_FORCE_WARNING":  "Interface Force Elevated — Warning",
    "ARM_POSITION_CRITICAL":    "Arm Position Error — Critical",
    "ARM_POSITION_WARNING":     "Arm Position Error — Warning",
}

_RECOMMENDED_ACTIONS: dict[str, str] = {
    "ATTITUDE_CRITICAL":        "Abort operation — stabilise attitude before retrying",
    "ATTITUDE_WARNING":         "Monitor attitude control; prepare abort if trend continues",
    "BUS_VOLTAGE_CRITICAL":     "Abort immediately — investigate power fault",
    "BUS_VOLTAGE_WARNING":      "Reduce non-essential loads; monitor bus voltage",
    "SEAL_PRESSURE_CRITICAL":   "Halt transfer — inspect interface seal integrity",
    "LEAK_DETECTED":            "Abort — vent line and inspect for breach",
    "LINE_PRESSURE_CRITICAL":   "Halt transfer — possible blockage or over-pressure fault",
    "LINE_PRESSURE_WARNING":    "Reduce flow rate; monitor upstream pressure trend",
    "PUMP_OVERCURRENT_CRITICAL":"Halt pump — inspect for blockage or pump failure",
    "PUMP_OVERCURRENT_WARNING": "Monitor pump health; reduce flow rate if trend continues",
    "INTERFACE_FORCE_CRITICAL": "Halt — risk of structural damage to docking interface",
    "INTERFACE_FORCE_WARNING":  "Verify arm alignment; reduce interface loads",
    "ARM_POSITION_CRITICAL":    "Abort docking — re-align end effector before contact",
    "ARM_POSITION_WARNING":     "Pause and re-check arm alignment before proceeding",
}


def group_alerts(
    alerts: list[RuleAlert],
    gap_s: float = 15.0,
) -> pd.DataFrame:
    """
    Merge timestamp-level rule alerts into event windows for dashboard display.

    Two alerts merge into the same event window when they share the same
    rule_name and phase, and the time gap between them is ≤ gap_s seconds.
    This collapses repeated firing from oscillating signals into one meaningful
    event rather than dozens of rows.

    Parameters
    ----------
    alerts : output of evaluate_rules() — unchanged
    gap_s  : maximum time gap (seconds) that still counts as the same event

    Returns
    -------
    DataFrame with one row per event window, sorted by start time then severity.
    Columns: rule_id, title, severity, phase, start_elapsed_s, end_elapsed_s,
             duration_s, description, recommended_action, affected_signals,
             peak_value, number_of_points
    """
    if not alerts:
        return pd.DataFrame(columns=[
            "rule_id", "title", "severity", "phase",
            "start_elapsed_s", "end_elapsed_s", "duration_s",
            "description", "recommended_action",
            "affected_signals", "peak_value", "number_of_points",
        ])

    # Group raw alerts by (rule_name, phase)
    by_key: dict[tuple[str, str], list[RuleAlert]] = {}
    for a in alerts:
        key = (a.rule_name, a.phase)
        by_key.setdefault(key, []).append(a)

    rows: list[dict] = []
    for (rule_name, phase), group in by_key.items():
        group.sort(key=lambda a: a.time)

        # Split into windows wherever the gap exceeds gap_s
        windows: list[list[RuleAlert]] = [[group[0]]]
        for alert in group[1:]:
            if alert.time - windows[-1][-1].time <= gap_s:
                windows[-1].append(alert)
            else:
                windows.append([alert])

        for window in windows:
            peak = max(abs(a.value) for a in window)
            rows.append({
                "rule_id":            rule_name,
                "title":              _RULE_TITLES.get(rule_name, rule_name),
                "severity":           window[0].severity,
                "phase":              phase,
                "start_elapsed_s":    window[0].time,
                "end_elapsed_s":      window[-1].time,
                "duration_s":         round(window[-1].time - window[0].time + 0.5, 1),
                "description":        window[0].description,
                "recommended_action": _RECOMMENDED_ACTIONS.get(rule_name, "Investigate"),
                "affected_signals":   window[0].signal,
                "peak_value":         round(peak, 3),
                "number_of_points":   len(window),
            })

    df = pd.DataFrame(rows)
    _sev_order = {"CRITICAL": 0, "WARNING": 1}
    df["_sev"] = df["severity"].map(_sev_order)
    df = (
        df.sort_values(["start_elapsed_s", "_sev"])
          .drop(columns=["_sev"])
          .reset_index(drop=True)
    )
    return df
