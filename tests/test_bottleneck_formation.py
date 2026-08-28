"""
Acceptance test from the sprint guide:
"force one station's mean processing time 25% above the rest with all
fault modes off. Confirm: that station has the highest utilization;
upstream stations show Blocked time; downstream stations show Starved time."
"""
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from plant.run import run_plant  # noqa: E402


def aggregate_state_time(states):
    """station -> {state: total_minutes}"""
    totals = defaultdict(lambda: defaultdict(float))
    for station, state, ts_start, ts_end in states:
        totals[station][state] += (ts_end - ts_start)
    return totals


def main():
    records, stations, build_seq, cfg = run_plant(
        "config/test_slowdown.yaml", "plant_out_test1", seed=7
    )

    slow_station = cfg["faults"]["constant_slowdown"]["station"]
    run_minutes = cfg["simulation"]["run_minutes"]

    print(f"Slowed station: {slow_station} (25% slower)\n")

    print("--- Utilization per station (busy_time / run_minutes) ---")
    util = {st.id: st.busy_time / run_minutes for st in stations}
    for sid in sorted(util):
        marker = "  <-- slowed station" if sid == slow_station else ""
        print(f"  station {sid:2d}: {util[sid]:.3f}{marker}")

    highest = max(util, key=util.get)
    print(f"\nHighest utilization: station {highest} "
          f"({'PASS' if highest == slow_station else 'FAIL'} - expected {slow_station})")

    state_totals = aggregate_state_time(records["states"])
    print("\n--- Blocked / Starved minutes per station ---")
    for sid in sorted(state_totals):
        blocked = state_totals[sid].get("Blocked", 0.0)
        starved = state_totals[sid].get("Starved", 0.0)
        tag = ""
        if sid == slow_station - 1:
            tag = "  <-- immediately upstream of slow station, expect high Blocked"
        elif sid == slow_station + 1:
            tag = "  <-- immediately downstream of slow station, expect high Starved"
        print(f"  station {sid:2d}: Blocked={blocked:8.1f} min  Starved={starved:8.1f} min{tag}")

    upstream = slow_station - 1
    downstream = slow_station + 1
    upstream_blocked = state_totals[upstream].get("Blocked", 0.0)
    downstream_starved = state_totals[downstream].get("Starved", 0.0)

    print(f"\nUpstream station {upstream} Blocked time: {upstream_blocked:.1f} min "
          f"({'PASS' if upstream_blocked > 100 else 'FAIL'} - expect substantial)")
    print(f"Downstream station {downstream} Starved time: {downstream_starved:.1f} min "
          f"({'PASS' if downstream_starved > 100 else 'FAIL'} - expect substantial)")


if __name__ == "__main__":
    main()
