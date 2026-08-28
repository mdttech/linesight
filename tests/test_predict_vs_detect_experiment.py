"""
The acceptance test for Phase 3, and the headline result of the whole
sprint: does acting on the PREDICTED bottleneck beat acting on the
DETECTED one, and does both beat doing nothing (FIFO)? Three arms, run
across several seeded replications, compared on throughput.

The predicted/detected arms both use the same discovered+tuned model,
built once from a reference run -- representing a twin already built from
historical data before deployment, not rediscovered live for every
decision.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import statistics
import pandas as pd

from plant.config_loader import load_config
from plant.variants import build_variant_multipliers
from plant.line import build_and_run
from twin.discovery.generate import generate_model
from twin.discovery.tune import tune_model
from twin.decide.controller import controller_factory


def run_arm(cfg_path, mode, seed, G_tuned=None, decision_interval=60,
            detection_window=60, prediction_horizon=25, boost_factor=0.85):
    cfg = load_config(cfg_path)
    rng = random.Random(seed)
    variant_multipliers = build_variant_multipliers(cfg)

    factory = None
    if mode in ("detected", "predicted"):
        factory = controller_factory(
            mode, G_tuned, decision_interval, detection_window,
            prediction_horizon, boost_factor, seed_base=seed)

    records, stations, build_seq = build_and_run(cfg, rng, variant_multipliers, factory)

    run_minutes = cfg["simulation"]["run_minutes"]
    n = cfg["n_stations"]
    completions = sum(1 for pid, st, ts, tf, r, sc in records["events"] if st == n)
    return completions / run_minutes


def build_reference_model(cfg_path, seed):
    """Discover+tune once from a reference run -- the twin as it would
    already exist before this experiment starts."""
    cfg = load_config(cfg_path)
    rng = random.Random(seed)
    variant_multipliers = build_variant_multipliers(cfg)
    records, stations, build_seq = build_and_run(cfg, rng, variant_multipliers)

    df = pd.DataFrame(records["events"],
                       columns=["part_id", "activity", "ts_start", "ts_finish", "result", "scrap"])
    variant_map = {row["part_id"]: row["variant"] for row in build_seq}
    G = generate_model(df, variant_map=variant_map)
    return tune_model(G, target_size=len(G.nodes))


def mean_ci(values):
    m = statistics.mean(values)
    if len(values) > 1:
        se = statistics.stdev(values) / (len(values) ** 0.5)
        return m, 1.96 * se
    return m, 0.0


def main(n_replications=8, run_minutes_override=None):
    cfg_path = "config/line_siteA.yaml"

    print("Building reference twin from a historical run (seed=42)...")
    G_tuned = build_reference_model(cfg_path, seed=42)
    print(f"  {len(G_tuned.nodes)} nodes, {len(G_tuned.edges)} edges discovered.\n")

    if run_minutes_override is not None:
        cfg = load_config(cfg_path)
        cfg["simulation"]["run_minutes"] = run_minutes_override
        import yaml
        with open("config/_experiment_tmp.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        cfg_path = "config/_experiment_tmp.yaml"

    results = {"fifo": [], "detected": [], "predicted": []}
    seeds = list(range(100, 100 + n_replications))

    for seed in seeds:
        results["fifo"].append(run_arm(cfg_path, "fifo", seed))
        results["detected"].append(run_arm(cfg_path, "detected", seed, G_tuned))
        results["predicted"].append(run_arm(cfg_path, "predicted", seed, G_tuned))
        print(f"seed {seed}: fifo={results['fifo'][-1]:.4f}  "
              f"detected={results['detected'][-1]:.4f}  "
              f"predicted={results['predicted'][-1]:.4f}")

    print(f"\n=== Results across {n_replications} replications ===")
    summary = {}
    for arm in ("fifo", "detected", "predicted"):
        m, ci = mean_ci(results[arm])
        summary[arm] = m
        print(f"  {arm:10s}: {m:.4f} +/- {ci:.4f} parts/min")

    print()
    pred_vs_fifo = 100 * (summary["predicted"] - summary["fifo"]) / summary["fifo"]
    pred_vs_det = 100 * (summary["predicted"] - summary["detected"]) / summary["detected"]
    print(f"Predicted vs FIFO:     {pred_vs_fifo:+.2f}%  "
          f"({'PASS' if summary['predicted'] > summary['fifo'] else 'FAIL'})")
    print(f"Predicted vs Detected: {pred_vs_det:+.2f}%  "
          f"({'better' if summary['predicted'] > summary['detected'] else 'not better'})")

    # Paired comparison -- each seed gives a matched fifo/detected/predicted
    # triple on the same random draws, so the per-seed difference is a
    # tighter, more honest signal than comparing the three arms' separate
    # (unpaired) confidence intervals against each other.
    print("\n--- Paired per-seed differences (removes seed-to-seed noise) ---")
    diffs_pf = [p - f for p, f in zip(results["predicted"], results["fifo"])]
    diffs_pd = [p - d for p, d in zip(results["predicted"], results["detected"])]
    m_pf, ci_pf = mean_ci(diffs_pf)
    m_pd, ci_pd = mean_ci(diffs_pd)
    wins_pf = sum(1 for d in diffs_pf if d > 0)
    wins_pd = sum(1 for d in diffs_pd if d > 0)
    print(f"  predicted - fifo:     {m_pf:+.4f} +/- {ci_pf:.4f} parts/min  "
          f"(predicted ahead in {wins_pf}/{len(diffs_pf)} replications)")
    print(f"  predicted - detected: {m_pd:+.4f} +/- {ci_pd:.4f} parts/min  "
          f"(predicted ahead in {wins_pd}/{len(diffs_pd)} replications)")

    return results, summary


if __name__ == "__main__":
    main()
