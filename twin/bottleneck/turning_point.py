"""
The turning point method: the current bottleneck is the station spending
the least time waiting on anyone else -- lowest Blocked+Starved in the
window. Upstream of the true constraint, stations pile up Blocked time
(finished work with nowhere to put it); downstream, they pile up Starved
time (waiting on the constraint to feed them). The constraint itself sits
at the turning point between the two trends, spending the least time in
either.
"""
from twin.bottleneck.active_period import _clip_segments


def blocked_starved_time(state_records, station, t0, t1):
    blocked = starved = 0.0
    for state, start, end in _clip_segments(state_records, station, t0, t1):
        if state == "Blocked":
            blocked += end - start
        elif state == "Starved":
            starved += end - start
    return blocked, starved


def turning_point_bottleneck(state_records, stations, t0, t1):
    """Returns (station_id, blocked, starved) for the station with the
    lowest Blocked+Starved time in the window."""
    scored = {s: blocked_starved_time(state_records, s, t0, t1) for s in stations}
    best = min(scored, key=lambda s: sum(scored[s]))
    return best, scored[best][0], scored[best][1]
