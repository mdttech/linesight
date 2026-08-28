import csv
from datetime import timedelta


def _fmt(anchor_dt, minutes):
    return (anchor_dt + timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")


def write_event_log(events, path, anchor_dt):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["part_id", "activity", "ts_start", "ts_finish", "result", "scrap"])
        for part_id, station, t_start, t_finish, result, scrap in events:
            w.writerow([part_id, station, _fmt(anchor_dt, t_start), _fmt(anchor_dt, t_finish), result, scrap])


def write_state_log(states, path, anchor_dt):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["station", "state", "ts_start", "ts_end"])
        for station, state, ts_start, ts_end in states:
            w.writerow([station, state, _fmt(anchor_dt, ts_start), _fmt(anchor_dt, ts_end)])
