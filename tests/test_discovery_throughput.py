"""
Acceptance test: run the discovered/tuned model standalone and compare its
throughput to the real plant's, over the same duration. Uses the no-faults
config -- this is validating the discovery mechanism's fidelity, not
whether faults are visible (test_faults.py already covers that).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import pandas as pd
from plant.run import run_plant
from twin.discovery.generate import generate_model
from twin.discovery.tune import tune_model
from twin.discovery.to_simpy import run_generated_model


def main():
    records, stations, build_seq, cfg = run_plant(
        "config/line_siteA_nofaults.yaml", "plant_out_throughput", seed=99
    )
    run_minutes = cfg["simulation"]["run_minutes"]
    takt_min = cfg["takt_seconds"] / 60.0

    # True throughput: parts that completed the last station in the window
    true_completions = sum(
        1 for _, station, _, t_finish, _, _ in records["events"]
        if station == cfg["n_stations"] and t_finish <= run_minutes
    )
    true_throughput = true_completions / run_minutes
    print(f"True plant throughput: {true_completions} parts / {run_minutes:.0f} min "
          f"= {true_throughput:.4f} parts/min")

    df = pd.read_csv("plant_out_throughput/event_log.csv", parse_dates=["ts_start", "ts_finish"])
    G = generate_model(df)
    G_tuned = tune_model(G, target_size=len(G.nodes))  # no-op at this size, run for completeness

    rng = random.Random(123)
    completions, gen_stations = run_generated_model(
        G_tuned, run_minutes=run_minutes, release_interval=takt_min, rng=rng
    )
    gen_throughput = len(completions) / run_minutes
    print(f"Generated model throughput: {len(completions)} parts / {run_minutes:.0f} min "
          f"= {gen_throughput:.4f} parts/min")

    pct_error = 100 * abs(gen_throughput - true_throughput) / true_throughput
    print(f"\nThroughput error: {pct_error:.1f}%")


if __name__ == "__main__":
    main()
