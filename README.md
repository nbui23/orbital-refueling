# Orbital Refueling Telemetry Simulator

Autonomous orbital refueling anomaly detection: a synthetic Python prototype demonstrating hybrid deterministic and machine-learning monitoring for spacecraft autonomy.

Orbital Refueling Telemetry Simulator is an educational portfolio project. It is not flight software, not an autonomous abort system, not a real spacecraft diagnosis tool, and not based on real spacecraft telemetry. All telemetry is procedurally generated with simplified physics.

## Why This Project Exists

Autonomous propellant transfer between spacecraft is a high-stakes operation with several subsystems moving at once: robotic alignment, docking contact loads, seal pressure, transfer pressure, pump behavior, fluid flow, attitude stability, thermal health, and electrical health.

Some faults are obvious because one signal crosses a hard engineering limit. Others are subtle: a pressure sensor can drift, a pump can degrade gradually, or flow and pressure can move in a correlated pattern that is individually borderline but collectively unusual.

Orbital Refueling Telemetry Simulator demonstrates a monitoring architecture for that split:

- deterministic rules for explicit hard-limit checks
- a phase-aware ML anomaly detector for advisory early warning
- lightweight feature attribution to explain which signals influenced the ML score
- a Streamlit dashboard for scenario exploration and validation

## Demo Highlights

- Full 410-second synthetic orbital refueling mission sampled at 2 Hz
- Nine mission phases, from approach through retreat
- Seven selectable scenarios: nominal plus six anomaly cases
- Deterministic rule alerts with severity and recommended actions
- One `IsolationForest` model per mission phase
- Perturbation-based signal contribution estimates
- Beginner Mode with plain-English chart captions and a signal glossary
- Scenario validation script for repeatable regression checks

## Why Hybrid Monitoring Matters

| Failure type | Example | Monitoring layer |
|---|---|---|
| Hard-limit breach | Pump current spikes above the safety threshold | Deterministic rule alert |
| Subtle multivariate drift | Flow, pressure, and pump current move into an unusual combined pattern | ML anomaly score |

Rules alone may miss gradual correlated degradation. ML alone is not appropriate as the authority for safety-critical decisions. Orbital Refueling Telemetry Simulator keeps both layers independent: rules provide hard engineering checks, while ML provides advisory pattern recognition.

## Refueling Phases

| Phase | Duration | What happens |
|---|---:|---|
| `approach` | 60 s | Servicer closes distance; attitude stabilizes |
| `arm_alignment` | 45 s | Robotic arm moves toward the target port |
| `docking` | 30 s | End effector makes contact; interface loads rise |
| `seal_check` | 20 s | Seal pressure ramps from 0 to 5 bar |
| `pressure_equalization` | 40 s | Transfer lines reach operating pressure |
| `main_transfer` | 120 s | Propellant flows at nominal 0.5 kg/s |
| `leak_check` | 30 s | Flow is halted; residual flow indicates a breach |
| `disconnect` | 20 s | Seal vents; arm retracts |
| `retreat` | 45 s | Servicer departs |

## Anomaly Scenarios

| Scenario | Injected behavior | Key signals |
|---|---|---|
| `nominal` | Clean baseline | None |
| `slow_leak` | Seal pressure drifts downward; leak-check flow rises | `seal_pressure`, `flow_rate` |
| `partial_blockage` | Flow drops while line pressure and pump current rise | `flow_rate`, `line_pressure`, `pump_current` |
| `pump_degradation` | Current spikes, flow becomes erratic, line heats up | `pump_current`, `flow_rate`, `line_temperature` |
| `arm_misalignment` | End-effector error stays high; interface loads rise | `end_effector_position_error`, `interface_force` |
| `sensor_drift` | Pressure readings drift upward over the mission | `line_pressure`, `donor_tank_pressure` |
| `unstable_slosh` | Flow and line pressure oscillate during transfer | `flow_rate`, `line_pressure`, `propellant_temperature` |

## Project Structure

| Path | Purpose |
|---|---|
| `app.py` | Streamlit dashboard |
| `simulator.py` | Synthetic mission telemetry and anomaly injection |
| `rules.py` | Deterministic engineering rules and grouped alerts |
| `detector.py` | Phase-aware `IsolationForest` anomaly detector |
| `explainer.py` | Perturbation-based feature attribution |
| `scripts/validate_scenarios.py` | Scenario validation summary |
| `tests/` | Regression tests for simulator, rules, detector, and explanations |
| `docs/` | Architecture and implementation notes |

## How To Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`. The detector trains on nominal synthetic data on first load and then reuses Streamlit's cache for scenario switches.

Requirements: Python 3.10+, no GPU needed.

## Run Tests

```bash
pip install -r requirements-dev.txt
python3 -m pytest tests/ -v
```

The test suite covers telemetry generation, rule logic, detector scoring, and explanation behavior.

Continuous integration runs the same test suite on Python 3.10, 3.11, and 3.12 through GitHub Actions.

## Run Scenario Validation

```bash
python3 scripts/validate_scenarios.py
```

The script trains or loads the default phase-aware detector, runs every scenario, prints a compact summary table, and writes `outputs/scenario_validation.csv`.

Key validation fields:

| Column | Meaning |
|---|---|
| `max_ml_score` | Highest phase-aware ML anomaly score observed |
| `mean_ml_score` | Average ML anomaly score across the mission |
| `ml_anomaly_rate_pct` | Percent of rows above the dashboard ML threshold |
| `raw_rule_alert_count` | Count of raw deterministic rule alert rows |
| `grouped_rule_alert_count` | Count of grouped dashboard alert events |
| `highest_rule_severity` | Highest deterministic severity: `NONE`, `WARNING`, or `CRITICAL` |
| `top_contributing_signal` | Largest perturbation-based contributor to high ML scores |

Expected validation story:

- `nominal` remains quiet, with low ML scores and no rule alerts.
- `partial_blockage` triggers both ML scoring and deterministic pressure/current rules.
- `sensor_drift` raises ML scores while deterministic rules can remain quiet.
- `unstable_slosh` raises ML scores through coupled flow, pressure, and thermal behavior without necessarily crossing hard thresholds.

These numbers are regression evidence for synthetic scenarios, not real-world performance estimates.

## Documentation

This repository keeps the public overview in `README.md` and `MODEL_CARD.md`. Extended local notes and walkthrough material are intentionally excluded from the public GitHub repository.

## Limitations

- Synthetic data only; the simulator is not a high-fidelity spacecraft model.
- Not flight software, an autonomous abort system, or a certified safety interlock.
- Rule thresholds are illustrative and not derived from real spacecraft engineering limits.
- `IsolationForest` scores each row independently and does not model temporal sequences.
- Perturbation attribution is approximate and not causal proof.
- The detector is trained offline and does not perform online learning or drift adaptation.

## Future Improvements

- Add sequence-aware anomaly detection such as an LSTM autoencoder or temporal convolutional model.
- Add state estimation with a Kalman filter or similar estimator.
- Add richer sensor fault scenarios such as dropout, stuck-at readings, and bias oscillation.
- Add confidence intervals around anomaly scores.
- Add online drift monitoring and retraining workflows.
- Add an ingestion layer for replaying external telemetry datasets.
