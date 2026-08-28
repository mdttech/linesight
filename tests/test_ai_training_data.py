"""
Acceptance test for the training-table builder: correctness of the fast
label computation against the trusted (but much slower) general-purpose
active_period method, and a speed check, since this needs to run across
several seeded 48h runs without becoming the bottleneck itself.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import numpy as np
from plant.run import run_plant
from twin.ai.build_training_table import build_training_table, _labels_from_momentary_bottleneck
from twin.bottleneck.active_period import momentary_bottleneck


def main():
    records, stations_obj, build_seq, cfg = run_plant(
        "config/line_siteA.yaml", "plant_out_ai_table_check", seed=42
    )
    station_ids = list(range(1, cfg["n_stations"] + 1))

    print("--- Correctness: fast label computation vs. trusted slow version ---")
    sample_times = np.arange(0, 200, 2.0)
    fast = _labels_from_momentary_bottleneck(records["states"], station_ids, sample_times, 20.0)
    mismatches = 0
    for t in sample_times:
        slow_winner, _ = momentary_bottleneck(records["states"], station_ids, t, t + 20.0)
        if fast[t] != slow_winner:
            mismatches += 1
    print(f"Mismatches: {mismatches} / {len(sample_times)}  "
          f"({'PASS' if mismatches == 0 else 'FAIL'})")

    print("\n--- Speed: full table build for one 48h run ---")
    tiers = {}
    for shop in cfg["shops"].values():
        for sid in shop["stations"]:
            tiers[sid] = shop["default_tier"]
        for sid_str, t in shop.get("overrides", {}).items():
            tiers[int(sid_str)] = t

    t0 = time.time()
    table = build_training_table(
        records, build_seq, station_ids, tiers, cfg["simulation"]["run_minutes"], run_id=42)
    elapsed = time.time() - t0
    print(f"Table shape: {table.shape}")
    print(f"Build time: {elapsed:.2f}s  ({'PASS' if elapsed < 5 else 'FAIL'} - expect well under 5s)")
    print(f"Positive rate: {table['label'].mean():.4f}")


if __name__ == "__main__":
    main()
