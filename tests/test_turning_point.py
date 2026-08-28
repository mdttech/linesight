"""
Acceptance test: the turning point method (lowest Blocked+Starved time)
agrees with the active period method on a known, deliberately-injected
bottleneck.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plant.run import run_plant
from twin.bottleneck.active_period import momentary_bottleneck
from twin.bottleneck.turning_point import turning_point_bottleneck


def main():
    records, stations, build_seq, cfg = run_plant(
        "config/test_slowdown.yaml", "plant_out_turning_point", seed=7
    )
    run_min = cfg["simulation"]["run_minutes"]
    station_ids = list(range(1, cfg["n_stations"] + 1))
    slow_station = cfg["faults"]["constant_slowdown"]["station"]

    s_ap, active_min = momentary_bottleneck(records["states"], station_ids, 0, run_min)
    s_tp, blocked, starved = turning_point_bottleneck(records["states"], station_ids, 0, run_min)

    print(f"Active period method:  station {s_ap} (active {active_min:.1f} min)")
    print(f"Turning point method:  station {s_tp} (blocked={blocked:.1f}, starved={starved:.1f})")
    print(f"\nExpected station {slow_station}:")
    print(f"  Active period: {'PASS' if s_ap == slow_station else 'FAIL'}")
    print(f"  Turning point: {'PASS' if s_tp == slow_station else 'FAIL'}")
    print(f"  Both agree with each other: {'PASS' if s_ap == s_tp else 'FAIL'}")


if __name__ == "__main__":
    main()
