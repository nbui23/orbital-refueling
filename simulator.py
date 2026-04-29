"""
simulator.py
Synthetic telemetry generator for autonomous orbital refueling operations.
All values are physically plausible but not based on real spacecraft data.
This is an educational prototype demonstrating AI/ML thinking for spacecraft autonomy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PHASES = [
    "approach",
    "arm_alignment",
    "docking",
    "seal_check",
    "pressure_equalization",
    "main_transfer",
    "leak_check",
    "disconnect",
    "retreat",
]

PHASE_DURATIONS: dict[str, int] = {  # seconds
    "approach": 60,
    "arm_alignment": 45,
    "docking": 30,
    "seal_check": 20,
    "pressure_equalization": 40,
    "main_transfer": 120,
    "leak_check": 30,
    "disconnect": 20,
    "retreat": 45,
}

SAMPLE_RATE: float = 2.0  # Hz (0.5 s timestep)

ANOMALY_SCENARIOS: list[str] = [
    "nominal",
    "slow_leak",
    "partial_blockage",
    "pump_degradation",
    "arm_misalignment",
    "sensor_drift",
    "sensor_dropout",
    "stuck_at_pressure",
    "bias_oscillation",
    "unstable_slosh",
]

SCENARIO_DESCRIPTIONS: dict[str, str] = {
    "nominal": "All systems nominal. No anomalies injected.",
    "slow_leak": "Seal pressure slowly drifts downward. Subtle flow detected during leak check.",
    "partial_blockage": "Transfer line partially blocked. Flow reduced; upstream pressure elevated; pump works harder.",
    "pump_degradation": "Pump efficiency degrades mid-transfer. Current spikes; flow rate becomes erratic.",
    "arm_misalignment": "End effector never fully converges to target. Interface forces elevated throughout docking.",
    "sensor_drift": "Line pressure sensor develops a slow positive bias. Readings diverge from true state.",
    "sensor_dropout": "Line pressure briefly drops to zero during transfer, creating obvious sensor-data gaps.",
    "stuck_at_pressure": "Line pressure sensor freezes at one reading during transfer while flow continues.",
    "bias_oscillation": "Line pressure sensor bias oscillates without crossing hard pressure limits.",
    "unstable_slosh": "Propellant slosh causes oscillations in flow rate and line pressure during transfer.",
}

# Nominal physical constants (synthetic reference values)
_DONOR_P: float = 300.0       # bar — donor tank initial pressure
_RECEIVER_P: float = 50.0     # bar — receiver tank initial pressure
_LINE_P: float = 150.0        # bar — nominal transfer line pressure
_SEAL_P: float = 5.0          # bar — nominal seal pressure
_BUS_V: float = 28.0          # V   — spacecraft bus voltage
_PROP_T: float = -20.0        # °C  — propellant temperature (MMH-like)
_LINE_T: float = -15.0        # °C  — transfer line temperature
_FLOW_NOM: float = 0.50       # kg/s — nominal flow rate


def _nominal_row(
    phase: str,
    t: float,
    t_phase: float,
    duration: float,
    rng: np.random.Generator,
    mass: float,
) -> dict:
    """Generate one nominally-correct telemetry row."""
    prog = t_phase / max(duration, 1.0)  # phase progress 0 → 1

    r: dict = {
        "phase": phase,
        "time": t,
        "system_mode": phase,
        "bus_voltage": rng.normal(_BUS_V, 0.1),
        "total_mass_transferred": mass,
    }

    # ── Attitude & Reaction Wheels ────────────────────────────────────────────
    if phase == "approach":
        # Start at 1.5° and converge to ~0.3° — stays below the 2.0° WARNING threshold
        r["attitude_error"] = max(0.01, rng.normal(1.5 - 1.2 * prog, 0.06))
        r["reaction_wheel_speed"] = rng.normal(1500 + 300 * prog, 20)
    elif phase in ("arm_alignment", "docking"):
        r["attitude_error"] = max(0.01, rng.normal(0.30, 0.02))
        r["reaction_wheel_speed"] = rng.normal(1800, 15)
    elif phase in ("pressure_equalization", "main_transfer"):
        r["attitude_error"] = max(0.01, rng.normal(0.40, 0.03))
        r["reaction_wheel_speed"] = rng.normal(1750, 20)
    else:
        r["attitude_error"] = max(0.01, rng.normal(0.25, 0.02))
        r["reaction_wheel_speed"] = rng.normal(1650, 15)

    # ── Arm ───────────────────────────────────────────────────────────────────
    if phase == "approach":
        r["arm_joint_angle"] = rng.normal(0.0, 0.3)
        r["arm_joint_velocity"] = rng.normal(0.0, 0.04)
        r["arm_motor_current"] = max(0.0, rng.normal(0.2, 0.05))
        r["end_effector_position_error"] = max(50.0, rng.normal(500 - 400 * prog, 10))

    elif phase == "arm_alignment":
        r["arm_joint_angle"] = rng.normal(45.0 * prog, 0.5)
        r["arm_joint_velocity"] = rng.normal(0.50, 0.05)
        r["arm_motor_current"] = rng.normal(2.5, 0.2)
        r["end_effector_position_error"] = max(1.0, rng.normal(50 - 48 * prog, 1.0))

    elif phase == "docking":
        r["arm_joint_angle"] = rng.normal(45.0, 0.2)
        r["arm_joint_velocity"] = rng.normal(0.05, 0.02)
        r["arm_motor_current"] = rng.normal(3.0, 0.2)
        r["end_effector_position_error"] = max(0.1, rng.normal(2.0 - 1.85 * prog, 0.2))

    elif phase in ("seal_check", "pressure_equalization", "main_transfer", "leak_check"):
        r["arm_joint_angle"] = rng.normal(45.0, 0.1)
        r["arm_joint_velocity"] = rng.normal(0.0, 0.01)
        r["arm_motor_current"] = rng.normal(1.5, 0.1)
        r["end_effector_position_error"] = max(0.05, rng.normal(0.20, 0.05))

    elif phase == "disconnect":
        r["arm_joint_angle"] = rng.normal(45.0 * (1 - prog), 0.3)
        r["arm_joint_velocity"] = rng.normal(-0.30, 0.05)
        r["arm_motor_current"] = rng.normal(2.0, 0.2)
        r["end_effector_position_error"] = max(0.1, rng.normal(0.2 + 30 * prog, 1.0))

    else:  # retreat
        r["arm_joint_angle"] = max(0.0, rng.normal(45.0 * (1 - prog), 0.5))
        r["arm_joint_velocity"] = rng.normal(-0.50, 0.05)
        r["arm_motor_current"] = rng.normal(2.5, 0.2)
        r["end_effector_position_error"] = rng.normal(50 + 450 * prog, 10)

    # ── Interface ─────────────────────────────────────────────────────────────
    if phase == "docking":
        r["interface_force"] = max(0.0, rng.normal(50 * prog, 3.0))
        r["interface_torque"] = max(0.0, rng.normal(5 * prog, 0.5))
    elif phase in ("seal_check", "pressure_equalization", "main_transfer", "leak_check"):
        r["interface_force"] = rng.normal(50.0, 2.0)
        r["interface_torque"] = rng.normal(5.0, 0.3)
    elif phase == "disconnect":
        r["interface_force"] = max(0.0, rng.normal(50 * (1 - prog), 2.0))
        r["interface_torque"] = max(0.0, rng.normal(5 * (1 - prog), 0.3))
    else:
        r["interface_force"] = max(0.0, rng.normal(0.0, 0.5))
        r["interface_torque"] = rng.normal(0.0, 0.1)

    # ── Seal ─────────────────────────────────────────────────────────────────
    if phase == "seal_check":
        r["seal_pressure"] = max(0.0, rng.normal(_SEAL_P * prog, 0.1))
    elif phase in ("pressure_equalization", "main_transfer", "leak_check"):
        r["seal_pressure"] = rng.normal(_SEAL_P, 0.05)
    elif phase == "disconnect":
        r["seal_pressure"] = max(0.0, rng.normal(_SEAL_P * (1 - prog), 0.1))
    else:
        r["seal_pressure"] = max(0.0, rng.normal(0.0, 0.02))

    # ── Tank / Fluid ──────────────────────────────────────────────────────────
    if phase in ("approach", "arm_alignment", "docking", "seal_check"):
        r["donor_tank_pressure"] = rng.normal(_DONOR_P, 1.0)
        r["receiver_tank_pressure"] = rng.normal(_RECEIVER_P, 0.5)
        r["line_pressure"] = max(0.0, rng.normal(0.0, 0.2))
        r["flow_rate"] = rng.normal(0.0, 0.005)
        r["pump_current"] = max(0.0, rng.normal(0.1, 0.05))

    elif phase == "pressure_equalization":
        eq = (_DONOR_P + _RECEIVER_P) / 2.0  # ~175 bar
        r["donor_tank_pressure"] = rng.normal(_DONOR_P - (_DONOR_P - eq) * prog, 1.0)
        r["receiver_tank_pressure"] = rng.normal(_RECEIVER_P + (eq - _RECEIVER_P) * prog, 0.5)
        r["line_pressure"] = max(0.0, rng.normal(_LINE_P * 0.5 * prog, 2.0))
        r["flow_rate"] = max(0.0, rng.normal(0.10 * prog, 0.01))
        r["pump_current"] = max(0.0, rng.normal(3.0 * prog, 0.3))

    elif phase == "main_transfer":
        r["donor_tank_pressure"] = rng.normal(_DONOR_P - mass * 0.40, 1.0)
        r["receiver_tank_pressure"] = rng.normal(_RECEIVER_P + mass * 0.80, 0.5)
        r["line_pressure"] = rng.normal(_LINE_P, 2.0)
        r["flow_rate"] = max(0.0, rng.normal(_FLOW_NOM, 0.02))
        r["pump_current"] = rng.normal(8.0, 0.3)

    elif phase == "leak_check":
        r["donor_tank_pressure"] = rng.normal(_DONOR_P - mass * 0.40, 0.5)
        r["receiver_tank_pressure"] = rng.normal(_RECEIVER_P + mass * 0.80, 0.3)
        r["line_pressure"] = rng.normal(_LINE_P, 1.0)
        r["flow_rate"] = rng.normal(0.0, 0.003)
        r["pump_current"] = max(0.0, rng.normal(0.5, 0.1))

    elif phase == "disconnect":
        r["donor_tank_pressure"] = rng.normal(_DONOR_P - mass * 0.40, 0.5)
        r["receiver_tank_pressure"] = rng.normal(_RECEIVER_P + mass * 0.80, 0.3)
        r["line_pressure"] = max(0.0, rng.normal(_LINE_P * (1 - prog), 2.0))
        r["flow_rate"] = rng.normal(0.0, 0.005)
        r["pump_current"] = max(0.0, rng.normal(0.1, 0.05))

    else:  # retreat
        r["donor_tank_pressure"] = rng.normal(_DONOR_P - mass * 0.40, 0.5)
        r["receiver_tank_pressure"] = rng.normal(_RECEIVER_P + mass * 0.80, 0.3)
        r["line_pressure"] = max(0.0, rng.normal(0.0, 0.2))
        r["flow_rate"] = rng.normal(0.0, 0.005)
        r["pump_current"] = max(0.0, rng.normal(0.1, 0.05))

    # ── Temperature ───────────────────────────────────────────────────────────
    if phase in ("pressure_equalization", "main_transfer"):
        r["propellant_temperature"] = rng.normal(_PROP_T + 2.0, 0.5)
        r["line_temperature"] = rng.normal(_LINE_T + 2.0, 0.5)
    else:
        r["propellant_temperature"] = rng.normal(_PROP_T, 0.3)
        r["line_temperature"] = rng.normal(_LINE_T, 0.3)

    return r


def _inject_anomaly(
    row: dict,
    scenario: str,
    phase: str,
    t: float,
    t_phase: float,
    t_anomaly_start: float,
    rng: np.random.Generator,
) -> None:
    """Modify row in-place to inject the chosen anomaly signal."""
    if scenario == "nominal":
        return

    elapsed = max(0.0, t - t_anomaly_start)

    if scenario == "slow_leak":
        # Seal pressure drifts downward; flow appears during leak_check
        if phase in ("pressure_equalization", "main_transfer", "leak_check", "disconnect"):
            row["seal_pressure"] = max(0.0, row["seal_pressure"] - 0.022 * elapsed)
            if row["line_pressure"] > 0:
                row["line_pressure"] = max(0.0, row["line_pressure"] - 0.06 * elapsed)
            if phase == "leak_check":
                row["flow_rate"] = max(0.0, row["flow_rate"] + rng.normal(0.028, 0.005))

    elif scenario == "partial_blockage":
        if phase in ("pressure_equalization", "main_transfer"):
            row["flow_rate"] = max(0.0, row["flow_rate"] * 0.40)
            row["line_pressure"] = row["line_pressure"] * 1.35
            row["pump_current"] = row["pump_current"] * 1.65

    elif scenario == "pump_degradation":
        if phase == "main_transfer":
            # Degradation ramps up over first 60 s of transfer
            deg = min(1.0, elapsed / 60.0)
            if deg > 0.25:
                spike = rng.choice([0, 1], p=[0.65, 0.35])
                row["pump_current"] += spike * rng.uniform(4.0, 9.0) * deg
                row["flow_rate"] = max(0.0, row["flow_rate"] * rng.uniform(0.50, 0.82))
                row["line_temperature"] += 7.0 * deg

    elif scenario == "arm_misalignment":
        if phase in (
            "arm_alignment", "docking", "seal_check",
            "pressure_equalization", "main_transfer", "leak_check",
        ):
            row["end_effector_position_error"] = max(28.0, rng.normal(38.0, 6.0))
            if phase not in ("arm_alignment",):
                row["interface_force"] += rng.normal(22.0, 7.0)
                row["interface_torque"] = row["interface_torque"] * 1.9 + rng.normal(0, 0.6)

    elif scenario == "sensor_drift":
        total = float(sum(PHASE_DURATIONS.values()))
        drift = 28.0 * (t / total)
        row["line_pressure"] = row["line_pressure"] + drift
        row["donor_tank_pressure"] = row["donor_tank_pressure"] + 12.0 * (t / total)

    elif scenario == "sensor_dropout":
        if phase == "main_transfer" and 35.0 <= t_phase <= 52.0:
            row["line_pressure"] = 0.0
        if phase == "leak_check" and 6.0 <= t_phase <= 10.0:
            row["receiver_tank_pressure"] = 0.0

    elif scenario == "stuck_at_pressure":
        if phase in ("main_transfer", "leak_check") and t_phase >= 25.0:
            row["line_pressure"] = _LINE_P + 1.5

    elif scenario == "bias_oscillation":
        if phase in ("pressure_equalization", "main_transfer", "leak_check"):
            osc = float(np.sin(2 * np.pi * 0.035 * t_phase))
            row["line_pressure"] = row["line_pressure"] + 18.0 * osc
            row["donor_tank_pressure"] = row["donor_tank_pressure"] + 4.0 * osc

    elif scenario == "unstable_slosh":
        if phase in ("pressure_equalization", "main_transfer"):
            osc_fast = float(np.sin(2 * np.pi * 0.10 * t_phase))
            osc_slow = float(np.sin(2 * np.pi * 0.05 * t_phase))
            row["flow_rate"] = max(0.0, row["flow_rate"] + 0.18 * osc_fast)
            row["line_pressure"] = row["line_pressure"] + 20.0 * osc_fast
            row["propellant_temperature"] = row["propellant_temperature"] + 1.8 * osc_slow


def generate_telemetry(scenario: str = "nominal", seed: int = 42) -> pd.DataFrame:
    """
    Generate a complete synthetic refueling telemetry sequence.

    Parameters
    ----------
    scenario : one of ANOMALY_SCENARIOS
    seed     : random seed for reproducibility

    Returns
    -------
    DataFrame with one row per timestep (0.5 s intervals), 21 telemetry columns.
    """
    if scenario not in ANOMALY_SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from: {ANOMALY_SCENARIOS}")

    rng = np.random.default_rng(seed)

    # Compute global time at which the anomaly-relevant phase begins
    _anomaly_phase_map: dict[str, str] = {
        "slow_leak": "pressure_equalization",
        "partial_blockage": "pressure_equalization",
        "pump_degradation": "main_transfer",
        "arm_misalignment": "arm_alignment",
        "sensor_drift": "approach",
        "sensor_dropout": "main_transfer",
        "stuck_at_pressure": "main_transfer",
        "bias_oscillation": "pressure_equalization",
        "unstable_slosh": "main_transfer",
    }
    anomaly_start_phase = _anomaly_phase_map.get(scenario, "approach")
    t_anomaly_start = 0.0
    for ph in PHASES:
        if ph == anomaly_start_phase:
            break
        t_anomaly_start += PHASE_DURATIONS[ph]

    rows: list[dict] = []
    t = 0.0
    mass = 0.0

    for phase in PHASES:
        duration = PHASE_DURATIONS[phase]
        n_steps = int(duration * SAMPLE_RATE)

        for step in range(n_steps):
            t_phase = step / SAMPLE_RATE
            row = _nominal_row(phase, t, t_phase, duration, rng, mass)
            _inject_anomaly(row, scenario, phase, t, t_phase, t_anomaly_start, rng)

            row["total_mass_transferred"] = mass
            if phase == "main_transfer":
                mass += max(0.0, row["flow_rate"]) / SAMPLE_RATE

            rows.append(row)
            t += 1.0 / SAMPLE_RATE

    df = pd.DataFrame(rows)

    # Enforce canonical column order
    ordered_cols = [
        "phase", "time",
        "arm_joint_angle", "arm_joint_velocity", "arm_motor_current",
        "end_effector_position_error", "interface_force", "interface_torque",
        "seal_pressure", "donor_tank_pressure", "receiver_tank_pressure",
        "line_pressure", "flow_rate", "total_mass_transferred", "pump_current",
        "propellant_temperature", "line_temperature",
        "attitude_error", "reaction_wheel_speed",
        "bus_voltage", "system_mode",
    ]
    return df[ordered_cols].reset_index(drop=True)
