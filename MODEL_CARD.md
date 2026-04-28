# Model Card — Orbital Refueling Telemetry Simulator Anomaly Detector

---

## Model Overview

| Field | Value |
|---|---|
| Model type | IsolationForest (scikit-learn), one instance per mission phase |
| Number of models | 9 (one per operation phase) |
| Input features | 17 numeric telemetry signals |
| Output | Anomaly score per timestep, range [0, 1] |
| Score interpretation | 0 = consistent with nominal; 1 = strongly inconsistent with nominal |
| Library | scikit-learn 1.x |
| Hyperparameters | n_estimators=200, contamination=0.02, random_state=42 |

---

## Training Data

**Source:** Synthetically generated telemetry from `simulator.py`.

**Generation:** Nominal scenario, seed=0, multiple runs. 820 timesteps × 17 features per run. Gaussian noise added to simulate sensor variability. No real spacecraft telemetry was used.

**What "nominal" means here:** Procedurally generated values consistent with the simulator's physics model (plausible magnitudes and phase transitions), with no anomaly injection. It does not represent any specific spacecraft, propellant, or mission profile.

**Contamination parameter:** 0.02 — the model tolerates approximately 2% of training rows as potential noise. This is a hyperparameter assumption, not a validated estimate of real sensor noise rates.

---

## Intended Use

**This model is intended for:**
- Demonstrating the concept of phase-aware multivariate anomaly detection for spacecraft telemetry
- Educational illustration of how ML and deterministic safety rules can be combined in a hybrid monitoring architecture
- Portfolio demonstration of AI/ML engineering thinking applied to autonomous systems

**This model is not intended for:**
- Flight software or onboard spacecraft systems
- Real-time safety decisions on actual spacecraft
- Autonomous abort decisions
- Replacing or supplementing certified safety interlocks
- Anomaly detection on real spacecraft telemetry without retraining and validation
- Diagnosing specific faults — the model produces an advisory score, not a diagnosis

---

## Score Normalization

Raw IsolationForest `decision_function()` output (threshold-relative score) is mapped to [0, 1] via:

```
score = 1 / (1 + exp(20 × decision_function_output))
```

- Values > 0 (inliers) map near 0
- Values < 0 (outliers) map near 1
- Value = 0 (decision boundary) maps to 0.5

The sigmoid steepness (k=20) is a fixed design choice, not validated against calibration data. The absolute score values should be treated as ordinal indicators, not calibrated probabilities.

---

## Performance on Synthetic Scenarios

Performance is measured on synthetic test data (seed=42), not real telemetry. These numbers reflect how well the model distinguishes synthetic anomalies from synthetic nominal data — they do not predict real-world performance.

| Scenario | Mean ML score | % timesteps > 0.45 | Rule alerts (grouped) |
|---|---|---|---|
| nominal | 0.238 | 4.8% | 0 |
| slow_leak | 0.338 | 26.5% | 2 |
| partial_blockage | 0.387 | 40.9% | 3 |
| pump_degradation | 0.343 | 28.7% | 2 |
| arm_misalignment | 0.437 | 52.4% | 10 |
| sensor_drift | 0.408 | 36.7% | 0 |
| unstable_slosh | 0.325 | 24.4% | 0 |

No precision/recall metrics are reported because this is a demonstration with synthetic ground truth and a single test seed. Formal evaluation would require a held-out dataset with independent generation seeds.

The key demonstration pattern is that `partial_blockage` triggers both ML and deterministic rules, while `sensor_drift` and `unstable_slosh` are ML-visible without deterministic rule alerts. This supports the hybrid-monitoring story, but it is not evidence of real spacecraft performance.

---

## Explanation Method

**Method:** Perturbation-based feature attribution (not SHAP).

For each anomalous timestep (score > 0.45), each feature is individually replaced with the nominal phase mean extracted from the StandardScaler fitted during training. The resulting score drop is that feature's attributed contribution.

**Limitations of this approach:**
- Assumes feature independence — attribution can be incorrect for correlated features
- Uses the scaler mean as "nominal," which is the training data mean, not a ground-truth nominal value
- Aggregation across timesteps may obscure within-phase variation
- Not equivalent to SHAP values; does not account for feature interaction effects
- Not causal proof and not a fault diagnosis

---

## Known Limitations

**Data limitations:**
- Training and test data are from the same simplified generative process (same simulator, different seeds). This measures distribution shift within synthetic data, not real-world generalization.
- Synthetic noise (Gaussian) does not capture real sensor failure modes: stuck-at faults, dropouts, quantization artifacts, bias temperature dependence.
- All anomaly scenarios are distinct and clean. Real faults often co-occur, evolve non-monotonically, or appear only in certain operating conditions.

**Model limitations:**
- IsolationForest treats each timestep independently. Temporal correlations (a signal rising over 30 seconds) are only captured indirectly through the feature values at each timestep, not through sequence modelling.
- Per-phase conditioning assumes phase labels are always correct. If phase detection fails (e.g., stuck in wrong phase state), the wrong model is applied and scores are unreliable.
- No uncertainty quantification. The model does not distinguish between "confidently nominal" and "uncertain" — both produce low scores.
- Contamination parameter (0.02) was not empirically validated against real nominal data distributions.

**Operational limitations:**
- Static model: no online learning, no drift detection, no recalibration mechanism.
- No complete failure-mode coverage. The model will not necessarily surface anomaly scenarios it was not tested against.
- Alert threshold (0.45) is a design choice, not tuned to a specific false-positive/false-negative operating point.
- Not flight software, not an autonomous abort system, and not a real spacecraft diagnosis system.

---

## Validation Required for Real Use

Before this architecture could be considered for any real application, the following would be needed:

1. **Real training data** — nominal telemetry from hardware-in-the-loop tests or prior flight operations, not synthetic data
2. **Rule threshold validation** — thresholds derived from spacecraft engineering documentation, reviewed by systems engineers and flight dynamics specialists
3. **Independent test set** — anomaly scenarios generated or recorded independently of training data, not from the same simulator
4. **False positive characterisation** — sustained nominal operations data to measure real false alarm rates under varying conditions
5. **Failure mode coverage analysis** — systematic enumeration of failure modes and assessment of which would and would not be detectable
6. **Integration testing** — latency, computational resource usage, and failure behaviour in the actual flight software environment
7. **Regulatory and safety review** — any safety-adjacent software in spacecraft operations would require formal review under applicable standards (e.g., ECSS, NASA-STD-8739)

---

## Intended Architectural Role

In a real deployment, this type of model would function as a **ground-side advisory monitoring layer**, not an onboard safety system. It would:

- Flag patterns for operator review
- Provide early warning to support mission planning decisions
- Never replace certified onboard safety interlocks or deterministic rule engines

The deterministic rule engine in this prototype represents what an onboard safety interlock would look like. The ML layer represents what a ground-based telemetry monitoring assistant might provide.

---

## Contact / Provenance

This model card describes a synthetic educational prototype. It was built to demonstrate AI/ML engineering thinking for autonomous spacecraft operations. No real spacecraft data, operational requirements, or proprietary engineering values were used.
