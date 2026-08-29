"""
A physics-informed sanity gate, not a training constraint: checks any
forecast against Little's Law (WIP = Throughput x Flow Time) before it
reaches a supervisor. Little's Law is classical queueing theory (Little,
1961) -- a conservation law, not a model to fit -- which is exactly why
this needed no training and cost about fifteen minutes to build.

'Flow time' here means total time a part spends in the system (entry to
exit), not a single station's processing time -- the two are easy to
conflate and only one of them is what Little's Law actually relates WIP
and throughput to.
"""


def little_law_consistency(wip, throughput, flow_time, tolerance=0.15):
    """wip: current work-in-progress count (parts in the system right now).
    throughput: parts/minute, recent observed rate.
    flow_time: average total minutes a part spends from entry to exit.
    Returns whether observed WIP is consistent with what Little's Law
    predicts from throughput and flow time, within tolerance."""
    predicted_wip = throughput * flow_time
    error = abs(predicted_wip - wip) / max(wip, 1e-6)
    return {
        "consistent": error <= tolerance,
        "error": error,
        "predicted_wip": predicted_wip,
        "observed_wip": wip,
    }


def current_wip(events, entry_activity, exit_activity, now):
    """Parts that have started the line (an event at entry_activity with
    ts_start <= now) but not yet finished it (no event at exit_activity
    with ts_finish <= now) -- the direct definition of WIP, computed the
    same log-only way state.py's buffer_wip is."""
    started = {pid for pid, st, t_start, t_finish, r, sc in events
               if st == entry_activity and t_start <= now}
    finished = {pid for pid, st, t_start, t_finish, r, sc in events
                if st == exit_activity and t_finish <= now}
    return len(started - finished)


def recent_throughput(events, exit_activity, now, window):
    """Parts/minute completed at exit_activity within [now-window, now]."""
    count = sum(1 for pid, st, t_start, t_finish, r, sc in events
                if st == exit_activity and now - window <= t_finish <= now)
    return count / window


def recent_flow_time(events, entry_activity, exit_activity, now, window):
    """Mean total time (exit finish - entry start) for parts that
    completed the line within [now-window, now]."""
    entry_start = {pid: t_start for pid, st, t_start, t_finish, r, sc in events
                   if st == entry_activity}
    flow_times = [
        t_finish - entry_start[pid]
        for pid, st, t_start, t_finish, r, sc in events
        if st == exit_activity and now - window <= t_finish <= now and pid in entry_start
    ]
    return sum(flow_times) / len(flow_times) if flow_times else 0.0
