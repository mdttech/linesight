"""
Builds a training table for the bottleneck-risk classifier: one row per
(station, timestamp) sample, features available at that instant, and a
label computed from what actually happened next in the log.

Labels are free here in a way they never are with real data: because this
is a completed historical run, the "future" the label depends on already
happened and is sitting right there in the log. (At inference time, on a
live plant, there is no future yet -- that's what Phase 3's roll-forward
simulation is for. Training doesn't need it; only inference does.)

Built for speed deliberately: naively recomputing buffer occupancy,
station state, and recent-cycle-time stats from scratch for every single
sample would be far too slow at ~1,400 samples/station/run x 12 stations
x several seeds. Instead, each of those is built once per run as a sorted
step function / rolling array and then looked up per sample with
np.searchsorted -- O(log n) per lookup instead of an O(n) rescan.
"""
import numpy as np
import pandas as pd

STATES = ["Running", "Down", "Blocked", "Starved"]
TIERS = ["instrumented", "partial", "manual"]


def _buffer_step_function(events, a, b, run_minutes):
    """Sorted (time, cumulative_occupancy) step function for the buffer
    between station a and b, from finish-at-a (+1) / start-at-b (-1)
    events -- same logic as Phase 2's buffer capacity discovery and
    Phase 3's state snapshot, just built once as a lookup table instead
    of recomputed per query."""
    finishes = [t_finish for pid, st, t_start, t_finish, r, sc in events if st == a]
    starts = [t_start for pid, st, t_start, t_finish, r, sc in events if st == b]
    times = np.array(finishes + starts, dtype=float)
    deltas = np.array([1] * len(finishes) + [-1] * len(starts), dtype=float)
    if len(times) == 0:
        return np.array([0.0]), np.array([0.0])
    order = np.argsort(times, kind="stable")
    times, deltas = times[order], deltas[order]
    cum = np.cumsum(deltas)
    return times, cum


def _lookup_step(times, values, query_times):
    idx = np.searchsorted(times, query_times, side="right") - 1
    idx = np.clip(idx, 0, len(values) - 1)
    result = values[idx]
    result[idx < 0] = 0.0
    return result


def _state_step_function(state_records, station):
    """Sorted (time, state) step function for one station's state, from
    its state_log rows (already start-ordered intervals)."""
    rows = sorted([(start, state) for s, state, start, end in state_records if s == station])
    if not rows:
        return np.array([0.0]), ["Starved"]
    times = np.array([r[0] for r in rows])
    states = [r[1] for r in rows]
    return times, states


def _lookup_state(times, states, query_times):
    idx = np.searchsorted(times, query_times, side="right") - 1
    idx = np.clip(idx, 0, len(states) - 1)
    return [states[i] for i in idx]


def _recent_cycle_stats(events, station, query_times, n=5):
    """For each query time, mean/std of the durations of the n most
    recently *completed* cycles at that station as of that time."""
    rows = sorted(
        [(t_finish, t_finish - t_start) for pid, st, t_start, t_finish, r, sc in events if st == station]
    )
    fin_times = np.array([r[0] for r in rows])
    durations = np.array([r[1] for r in rows])

    means, stds = [], []
    for qt in query_times:
        idx = np.searchsorted(fin_times, qt, side="right")
        window = durations[max(0, idx - n):idx]
        if len(window) == 0:
            means.append(0.0)
            stds.append(0.0)
        else:
            means.append(float(window.mean()))
            stds.append(float(window.std()) if len(window) > 1 else 0.0)
    return np.array(means), np.array(stds)


def _upcoming_shares(build_seq, query_times, lookahead=6):
    """For each query time, the fraction of Loaded/Mid variants among the
    next `lookahead` scheduled vehicles from that point on."""
    seq_times = np.array([r["planned_release_minute"] for r in build_seq])
    seq_variant = np.array([r["variant"] for r in build_seq])

    loaded_shares, mid_shares = [], []
    for qt in query_times:
        idx = np.searchsorted(seq_times, qt, side="left")
        window = seq_variant[idx:idx + lookahead]
        if len(window) == 0:
            loaded_shares.append(0.0)
            mid_shares.append(0.0)
        else:
            loaded_shares.append(float((window == "Loaded").mean()))
            mid_shares.append(float((window == "Mid").mean()))
    return np.array(loaded_shares), np.array(mid_shares)


def _labels_from_momentary_bottleneck(state_records, stations, sample_times, horizon):
    """Same result as calling active_period.momentary_bottleneck() once
    per sample time, but built for speed: state records are grouped and
    sorted by station once (O(n log n)), then each window query only
    touches the small slice of that station's records which could
    possibly overlap [t, t+horizon] via binary search -- instead of
    rescanning all ~2,000 records per station per query, which is what
    made the naive version take ~11s for a single run's table."""
    from collections import defaultdict

    by_station = defaultdict(list)
    for s, state, start, end in state_records:
        by_station[s].append((start, end, state))
    starts_by_station, ends_by_station = {}, {}
    for s in stations:
        recs = sorted(by_station[s], key=lambda r: r[0])
        by_station[s] = recs
        starts_by_station[s] = np.array([r[0] for r in recs])
        ends_by_station[s] = np.array([r[1] for r in recs])

    ACTIVE = {"Running", "Down"}

    def longest_active_fast(station, t0, t1):
        recs = by_station[station]
        starts = starts_by_station[station]
        # first record that could possibly end after t0
        lo = np.searchsorted(ends_by_station[station], t0, side="left")
        best = cur = 0.0
        for start, end, state in recs[lo:]:
            if start >= t1:
                break
            if end <= t0:
                continue
            seg_start, seg_end = max(start, t0), min(end, t1)
            if state in ACTIVE:
                cur += seg_end - seg_start
                best = max(best, cur)
            else:
                cur = 0.0
        return best

    winner_by_time = {}
    for t in sample_times:
        scores = {s: longest_active_fast(s, t, t + horizon) for s in stations}
        winner_by_time[t] = max(scores, key=scores.get)
    return winner_by_time


def build_training_table(records, build_seq, stations, tiers, run_minutes,
                          sample_interval=2.0, horizon=20.0, lookahead_vehicles=6,
                          run_id=0):
    sample_times = np.arange(0.0, run_minutes - horizon, sample_interval)
    winner_by_time = _labels_from_momentary_bottleneck(
        records["states"], stations, sample_times, horizon)

    rows = []
    for station in stations:
        buf_in_times, buf_in_vals = (np.array([0.0]), np.array([0.0]))
        idx = stations.index(station)
        if idx > 0:
            buf_in_times, buf_in_vals = _buffer_step_function(
                records["events"], stations[idx - 1], station, run_minutes)
        buf_out_times, buf_out_vals = (np.array([0.0]), np.array([0.0]))
        if idx < len(stations) - 1:
            buf_out_times, buf_out_vals = _buffer_step_function(
                records["events"], station, stations[idx + 1], run_minutes)

        state_times, state_vals = _state_step_function(records["states"], station)

        buf_in = _lookup_step(buf_in_times, buf_in_vals, sample_times)
        buf_out = _lookup_step(buf_out_times, buf_out_vals, sample_times)
        cur_state = _lookup_state(state_times, state_vals, sample_times)
        cyc_mean, cyc_std = _recent_cycle_stats(records["events"], station, sample_times)
        up_loaded, up_mid = _upcoming_shares(build_seq, sample_times, lookahead_vehicles)

        tier = tiers[station]
        labels = [1 if winner_by_time[t] == station else 0 for t in sample_times]

        df = pd.DataFrame({
            "run_id": run_id,
            "station": station,
            "t": sample_times,
            "buffer_in": buf_in,
            "buffer_out": buf_out,
            "current_state": cur_state,
            "recent_cycle_time_mean": cyc_mean,
            "recent_cycle_time_std": cyc_std,
            "sensor_tier": tier,
            "upcoming_loaded_share": up_loaded,
            "upcoming_mid_share": up_mid,
            "label": labels,
        })
        rows.append(df)

    table = pd.concat(rows, ignore_index=True)
    for s in STATES:
        table[f"state_{s}"] = (table["current_state"] == s).astype(int)
    for tr in TIERS:
        table[f"tier_{tr}"] = (table["sensor_tier"] == tr).astype(int)
    return table


FEATURE_COLUMNS = (
    ["buffer_in", "buffer_out", "recent_cycle_time_mean", "recent_cycle_time_std",
     "upcoming_loaded_share", "upcoming_mid_share"]
    + [f"state_{s}" for s in STATES]
    + [f"tier_{t}" for t in TIERS]
)
