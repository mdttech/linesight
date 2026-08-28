"""
Acceptance test from the sprint guide:
"turn faults back on and run 48h again -- confirm the wear station shows a
rising trend in cycle time, and the night-shift stations show increased
variance specifically during the configured hours."
"""
import sys
import os
import csv
from collections import defaultdict
from datetime import datetime
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from plant.run import run_plant  # noqa: E402


def main():
    records, stations, build_seq, cfg = run_plant(
        "config/line_siteA.yaml", "plant_out_test2", seed=42
    )

    wear = cfg["faults"]["equipment_wear"]
    ov = cfg["faults"]["operator_variation"]

    # --- equipment wear: rising cycle time trend at the wear station ---
    wear_station = wear["station"]
    cycle_times = []  # (t_start, duration)
    for part_id, station, t_start, t_finish, result, scrap in records["events"]:
        if station == wear_station:
            cycle_times.append((t_start, t_finish - t_start))
    cycle_times.sort()

    before = [d for t, d in cycle_times if t < wear["start_minute"]]
    after_early = [d for t, d in cycle_times if wear["start_minute"] < t < wear["start_minute"] + 600]
    after_late = [d for t, d in cycle_times if t > cfg["simulation"]["run_minutes"] - 300]

    print(f"Equipment wear at station {wear_station}, starting minute {wear['start_minute']}\n")
    print(f"  mean cycle time before wear starts:      {statistics.mean(before):.3f} min  (n={len(before)})")
    print(f"  mean cycle time shortly after wear starts: {statistics.mean(after_early):.3f} min  (n={len(after_early)})")
    print(f"  mean cycle time near end of run:          {statistics.mean(after_late):.3f} min  (n={len(after_late)})")
    rising = statistics.mean(after_late) > statistics.mean(after_early) > statistics.mean(before) * 0.98
    print(f"  Rising trend: {'PASS' if rising else 'FAIL'}\n")

    # --- operator variation: higher variance at night, at flagged stations ---
    print(f"Operator variation at stations {ov['stations']}, night hours {ov['night_shift_hours']}\n")
    for sid in ov["stations"]:
        day_durs, night_durs = [], []
        for part_id, station, t_start, t_finish, result, scrap in records["events"]:
            if station != sid:
                continue
            hour = int((t_start // 60) % 24)
            start_h, end_h = ov["night_shift_hours"]
            is_night = (hour >= start_h or hour < end_h) if start_h > end_h else (start_h <= hour < end_h)
            (night_durs if is_night else day_durs).append(t_finish - t_start)

        day_std = statistics.pstdev(day_durs) if len(day_durs) > 1 else 0
        night_std = statistics.pstdev(night_durs) if len(night_durs) > 1 else 0
        print(f"  station {sid}: day std={day_std:.4f} (n={len(day_durs)})  "
              f"night std={night_std:.4f} (n={len(night_durs)})  "
              f"{'PASS' if night_std > day_std else 'FAIL'}")


if __name__ == "__main__":
    main()
