"""
Sanity test for the sync and roll-forward mechanism: snapshot the plant's
real state mid-run, roll the discovered model forward from it, and confirm
the predicted bottleneck lands on or adjacent to the station actually
under stress (the equipment-wear station and its immediate neighbours) --
not somewhere unrelated on the line.

This is a sanity check, not a precision benchmark: a single 25-minute
window's exact station-by-station match against reality is inherently
noisy (see the Complete guide's notes on this). The real acceptance test
for the prediction mechanism is the throughput experiment
(test_predict_vs_detect_experiment.py), which aggregates across many
windows and replications.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from plant.run import run_plant
from twin.discovery.generate import generate_model
from twin.discovery.tune import tune_model
from twin.sync.state import snapshot
from twin.sync.rollforward import roll_forward
from twin.bottleneck.active_period import momentary_bottleneck, longest_active_period


def main():
    records, stations, build_seq, cfg = run_plant(
        "config/line_siteA.yaml", "plant_out_rollforward_check", seed=42
    )
    station_ids = list(range(1, cfg["n_stations"] + 1))
    wear_station = cfg["faults"]["equipment_wear"]["station"]
    variant_map = {row["part_id"]: row["variant"] for row in build_seq}

    df = pd.read_csv("plant_out_rollforward_check/event_log.csv",
                      parse_dates=["ts_start", "ts_finish"])
    G = generate_model(df, variant_map=variant_map)
    G_tuned = tune_model(G, target_size=len(G.nodes))

    now = 1500.0
    horizon = 25
    snap = snapshot(records, build_seq, station_ids, now)
    upcoming = [r for r in build_seq if r["planned_release_minute"] >= now][:30]

    sim_records = roll_forward(G_tuned, station_ids, snap, upcoming, now, horizon, seed=1)

    print(f"{'station':>8} {'real active_min':>18} {'predicted active_min':>22}")
    for s in station_ids:
        real = longest_active_period(records["states"], s, now, now + horizon)
        pred = longest_active_period(sim_records, s, now, now + horizon)
        marker = "  <-- wear station" if s == wear_station else ""
        print(f"{s:>8} {real:>18.2f} {pred:>22.2f}{marker}")

    real_bn, _ = momentary_bottleneck(records["states"], station_ids, now, now + horizon)
    pred_bn, _ = momentary_bottleneck(sim_records, station_ids, now, now + horizon)
    print(f"\nReal momentary bottleneck this window: station {real_bn}")
    print(f"Predicted bottleneck: station {pred_bn}")
    print(f"Equipment wear fault is at station {wear_station}")
    close = abs(pred_bn - wear_station) <= 1
    print(f"Predicted bottleneck is at or adjacent to the wear station: "
          f"{'PASS' if close else 'FAIL'}")


if __name__ == "__main__":
    main()
