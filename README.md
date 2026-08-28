# LineSight

**A digital twin that builds itself from data the plant already collects — and predicts the shifting bottleneck by rolling forward over a build sequence that's already known.**

Accenture Innovation Challenge 2026 · DigitalTwin.ai · Team IITK-AEUAS (Bhaskar Rajaura, Tahseen Aslam), IIT Kanpur

---

## The problem

On a vehicle assembly line, two failures compound quietly. Line throughput equals the throughput of its slowest station — but on a mixed-model line the constraint shifts as high-option vehicles cluster in the build sequence, and dashboards report this only after output is already lost. Separately, a defect introduced at a manual station goes undetected until end-of-line testing, hours downstream, by which point every vehicle built in between carries the same risk.

Both failures share one cause: the line can observe its current state but cannot anticipate its next one. That gap is hardest to close exactly where it matters most — general assembly stations are manual and largely un-instrumented, while body-shop stations are robot-dense and richly monitored.

## What LineSight does

- **Builds itself** — process-mines a plant's MES event log directly into a graph model, with no manual modelling.
- **Models uneven instrumentation explicitly** — every station carries a sensor tier (`instrumented` / `partial` / `manual`) rather than assuming a uniformly-monitored ideal factory.
- **Predicts by rolling forward** — synchronises to live state, then simulates ahead over the vehicle build sequence, which is scheduled and known in advance.
- **Reports honestly** — ranked candidate causes with confidence, not false certainty; explicit flags wherever it's inferring rather than observing.

## Architecture

```
plant/    the "physical" system. SimPy. Emits logs. Nothing else may import it.
twin/     everything actually being proposed. Reads logs only.
```

This separation is deliberate and enforced by design, not just convention. `plant/` stands in for a real assembly line — a calibrated simulator playing the role of what Ait-Alla et al. and Ragazzini et al. (2024) call the **Physical Twin**, the same construct used to generate the dataset in Waseem et al.'s General Motors study (*J. Manuf. Syst.* 86, 2026). `twin/` is the actual proposal: it consumes only `event_log.csv` and `state_log.csv`, never anything internal to the simulator — exactly the constraint a real deployment would face too.

## Status

| Layer | Status |
|---|---|
| **Plant simulator** — 12 stations, 3 shop zones, sensor tiers, 2 fault modes, reliability model | ✅ Built & verified |
| **L1 — self-building twin** (process-mine event log → graph → tuned model) | ✅ Built & verified |
| L3 — sync, bottleneck detection, roll-forward prediction | ⬜ Next |
| L4 — explainable bottleneck-risk classifier (XGBoost + SHAP) | ⬜ Planned |
| Defect-risk classifier (Bosch dataset) | ⬜ Planned |
| Read-only integration adapter, prediction ledger, three-tier UI | ⬜ Planned |
| Vision-based coverage of un-instrumented stations | ⬜ Designed, not built this round |

## Quickstart

```bash
python -m plant.run --config config/line_siteA.yaml --out plant_out --seed 42
python tests/test_bottleneck_formation.py
python tests/test_faults.py
python tests/test_discovery_accuracy.py
python tests/test_tune_unit.py
python tests/test_discovery_throughput.py
python tests/test_discovery_reconfigure.py
```

Every run is seeded — the same seed on the same code reproduces identical output, including the exact numbers below.

## Results

| Test | Result |
|---|---|
| Bottleneck formation (one station forced 25% slower) | Correctly identified as the constraint: 99.8% utilization vs. ~80% baseline; upstream Blocked 566.5 min, downstream Starved 586.7 min |
| Equipment wear fault | Cycle time rises 1.10 → 1.29 → 2.76 min over the run, as designed |
| Operator variation fault | Night-shift variance measurably higher than day at both flagged stations |
| **L1 discovery — node precision/recall** | **1.000 / 1.000** |
| **L1 discovery — arc precision/recall** | **1.000 / 1.000** |
| L1 discovery — buffer capacity MAE | 1.00 (see note below) |
| Tuning mechanism (isolated unit test) | Correctly merges the lowest-frequency node; correctly no-ops when already at target size |
| Generated model throughput vs. true plant | 12.6% error (see note below) |
| Reconfigure-and-regenerate (add a station, re-run discovery) | 12→13 nodes, 11→12 edges, zero discovery code touched |

The graph structure is recovered **exactly** from nothing but the event log — every node, every edge, no manual modelling.

### Two honest findings, not smoothed over

**Buffer capacity is off by exactly +1, on every single edge, always.** A station that finishes a part but is blocked from releasing it (downstream buffer full) gets recorded as "entered the buffer" at its finish timestamp — before it has actually landed in the buffer object. This is a real, bounded, fully-understood characteristic of estimating buffer occupancy from MES-style timestamps rather than internal system telemetry — exactly the constraint a real deployment faces, since a real system doesn't get privileged access to a factory's internal buffer state either.

**The regenerated model's throughput runs 12.6% higher than the true plant's.** The discovered processing-time distribution is resampled from `(ts_finish − ts_start)` — pure processing time, which correctly excludes station downtime since failures are modelled as occurring before a part is picked up, not during processing. The regenerated model doesn't yet have its own reliability model, so it runs closer to nominal capacity than the true plant does. Station availability is fully recoverable from `state_log.csv` — this is a well-scoped next step, not attempted this round.

## What's next

Rolling the discovered model forward over the known build sequence to predict — not just detect — the shifting bottleneck (L3), then a trained, explainable bottleneck-risk classifier on top of it (L4), following the same build-and-verify approach used above.

## References

Lugaresi, G. & Matta, A. (2021). Automated manufacturing system discovery and digital twin generation. *Journal of Manufacturing Systems*, 59, 51–66.

Ragazzini, L., Negri, E., Fumagalli, L. & Macchi, M. (2024). Digital Twin-based bottleneck prediction for improved production control. *Computers & Industrial Engineering*, 192, 110231.

Kumbhar, M., Ng, A.H.C. & Bandaru, S. (2023). A digital twin based framework for detection, diagnosis, and improvement of throughput bottlenecks. *Journal of Manufacturing Systems*, 66, 92–106.

Waseem, M., Tan, C., Oh, S.-C., Arinez, J., Zhou, Z. & Chang, Q. (2026). Spatio-temporal graph neural network based digital twin surrogate for throughput estimation in general assembly lines. *Journal of Manufacturing Systems*, 86, 641–647.

Selvaraj, V., Al-Amin, M., Yu, X., Tao, W. & Min, S. (2024). Real-time action localization of manual assembly operations using deep learning and augmented inference state machines. *Journal of Manufacturing Systems*, 72, 504–518.

Iyer, S.V., Sangwan, K.S. & Dhiraj (2025). A cognitive digital twin for process chain anomaly detection and bottleneck analysis. *Journal of Industrial and Production Engineering*, 42(1).

Yang, et al. (2025). Leveraging Large Language Models for Enhanced Digital Twin Modeling: Trends, Methods, and Challenges. arXiv:2503.02167.
