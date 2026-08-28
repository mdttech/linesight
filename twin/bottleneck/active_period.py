"""
The active period method (Roser et al.): the bottleneck is the station
with the longest uninterrupted run of active states -- Running or Down,
the states that can make a neighbour wait -- within a window. Run on a
past window of real state data, this is detection. Run on a simulated
future window (twin/sync/rollforward.py), this is prediction, using
exactly the same function.
"""

ACTIVE = {"Running", "Down"}
INACTIVE = {"Blocked", "Starved"}


def _clip_segments(state_records, station, t0, t1):
    """Yields (state, start, end) segments for one station, clipped to
    [t0, t1]. state_records: iterable of (station, state, ts_start, ts_end)."""
    for s, state, start, end in state_records:
        if s != station:
            continue
        if end <= t0 or start >= t1:
            continue
        yield state, max(start, t0), min(end, t1)


def longest_active_period(state_records, station, t0, t1):
    """Longest uninterrupted total time spent in ACTIVE states within the
    window, in chronological order. Segments are clipped to the window
    first and then sorted, since raw records aren't guaranteed ordered."""
    segments = sorted(_clip_segments(state_records, station, t0, t1), key=lambda x: x[1])
    best = cur = 0.0
    for state, start, end in segments:
        if state in ACTIVE:
            cur += end - start
            best = max(best, cur)
        else:
            cur = 0.0
    return best


def momentary_bottleneck(state_records, stations, t0, t1):
    """The station with the longest active period in the window -- the
    momentary bottleneck. Returns (station_id, active_minutes)."""
    scored = {s: longest_active_period(state_records, s, t0, t1) for s in stations}
    best_station = max(scored, key=scored.get)
    return best_station, scored[best_station]
