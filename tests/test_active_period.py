"""
Acceptance test: the active period method correctly identifies a known,
deliberately-injected bottleneck -- both over the full run and over a
narrower late-run window.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plant.run import run_plant
from twin.bottleneck.active_period import momentary_bottleneck


def main():
    records, stations, build_seq, cfg = run_plant(
        "config/test_slowdown.yaml", "plant_out_active_period", seed=7
    )
    run_min = cfg["simulation"]["run_minutes"]
    station_ids = list(range(1, cfg["n_stations"] + 1))
    slow_station = cfg["faults"]["constant_slowdown"]["station"]

    station, active_min = momentary_bottleneck(records["states"], station_ids, 0, run_min)
    print(f"Full-run momentary bottleneck: station {station} (active {active_min:.1f} min)")
    print(f"Expected station {slow_station}: {'PASS' if station == slow_station else 'FAIL'}")

    station2, active_min2 = momentary_bottleneck(
        records["states"], station_ids, run_min - 500, run_min)
    print(f"\nLast-500-min momentary bottleneck: station {station2} (active {active_min2:.1f} min)")
    print(f"Expected station {slow_station}: {'PASS' if station2 == slow_station else 'FAIL'}")


if __name__ == "__main__":
    main()
