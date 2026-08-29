"""
Statistical Process Control: Individuals-Moving Range (I-MR) charts, the
right chart for this data since we have one cycle-time value per part per
station, not natural subgroups (which is what X-bar/R charts assume).

This is the fast, cheap, interpretable first line of anomaly detection --
no training data, no model, just arithmetic. It runs upstream of the
trained classifier (Phase 4): a Western Electric flag is a fast tripwire
that can trigger the heavier model to run out-of-cycle rather than
waiting for its next scheduled pass.

Design point that matters and is easy to get wrong: control limits must
be established from a clean BASELINE period and then held FIXED while
monitoring later data against them (classical Phase I / Phase II SPC).
Computing limits from the *entire* series -- baseline and fault period
together -- lets the limits drift wide enough to absorb the fault itself,
which silently defeats the whole point. Verified empirically: an earlier
version of this file that used the whole series produced its first flag
227 minutes *before* the fault even started, and 2,730 flags across 1,515
cycles -- it was flagging almost everything, not the fault specifically.
"""
import numpy as np


def i_mr_limits(values, k=2.66):
    """k=2.66 is the standard I-MR constant (Montgomery, Introduction to
    Statistical Quality Control) for 3-sigma limits derived from the mean
    moving range, not a directly-computed standard deviation -- the
    classical I-MR convention."""
    values = np.asarray(values, dtype=float)
    mr = np.abs(np.diff(values))
    mr_bar = mr.mean() if len(mr) else 0.0
    x_bar = values.mean() if len(values) else 0.0
    sigma = mr_bar / 1.128  # d2 constant for n=2 moving ranges
    return {
        "centerline": x_bar,
        "ucl": x_bar + k * mr_bar,
        "lcl": x_bar - k * mr_bar,
        "mr_ucl": 3.267 * mr_bar,
        "sigma": sigma,
    }


def western_electric_flags(values, limits):
    """The four classic Western Electric rules (Western Electric Company,
    Statistical Quality Control Handbook, 1956), applied point-by-point
    against FIXED limits (from i_mr_limits, computed on baseline data only):

      Rule 1: one point beyond 3-sigma
      Rule 2: 2 of 3 consecutive points beyond 2-sigma, same side
      Rule 3: 4 of 5 consecutive points beyond 1-sigma, same side
      Rule 4: 8 consecutive points on one side of the centerline

    Returns a sorted list of (index, rule_number) for every flagged point
    in `values` -- a point can trigger more than one rule.
    """
    values = np.asarray(values, dtype=float)
    center = limits["centerline"]
    sigma = limits["sigma"]
    if sigma == 0:
        return []

    z = (values - center) / sigma
    flags = []

    for i in np.where(np.abs(z) > 3)[0]:
        flags.append((int(i), 1))

    for i in range(2, len(z)):
        window = z[i - 2:i + 1]
        if (window > 2).sum() >= 2 or (window < -2).sum() >= 2:
            flags.append((i, 2))

    for i in range(4, len(z)):
        window = z[i - 4:i + 1]
        if (window > 1).sum() >= 4 or (window < -1).sum() >= 4:
            flags.append((i, 3))

    for i in range(7, len(z)):
        window = z[i - 7:i + 1]
        if (window > 0).all() or (window < 0).all():
            flags.append((i, 4))

    return sorted(set(flags))


def station_cycle_times(event_records, station):
    """(times, durations) for one station, in completion order --
    ts_finish - ts_start per part, which is available for every station
    regardless of sensor tier. This is the one thing worth being precise
    about: 'manual' tier (per sensor_tiers.yaml) means no torque/quality
    sensor -- it does NOT mean no cycle-time signal, since even a manual
    station's MES check-in/check-out timestamps exist (the same reasoning
    Decision 2 already established for L1 discovery). SPC on cycle time
    runs everywhere; it's richer per-part signals like torque that
    wouldn't be available at a manual-tier station in reality."""
    rows = sorted(
        (t_finish, t_finish - t_start)
        for pid, st, t_start, t_finish, r, sc in event_records if st == station
    )
    times = [r[0] for r in rows]
    durations = [r[1] for r in rows]
    return times, durations


def detect(event_records, station, baseline_start=60.0, baseline_end=200.0):
    """End-to-end, with the baseline/monitoring split that actually makes
    this a fault-detection tool rather than a self-defeating one. Default
    baseline window (60-200 min) is a stable early-operation period,
    chosen as a general 'settled in, nothing unusual yet' reference --
    not reverse-engineered to any specific fault's start time, though for
    line_siteA.yaml's default config it comfortably precedes the
    equipment_wear fault at minute 240.

    Returns (times, durations, limits, flags) where flags are
    (time, rule) for every flagged point in the MONITORING period only
    -- the baseline period is never flagged against itself.
    """
    times, durations = station_cycle_times(event_records, station)
    times, durations = np.array(times), np.array(durations)

    baseline_mask = (times >= baseline_start) & (times < baseline_end)
    monitor_mask = times >= baseline_end

    baseline_durations = durations[baseline_mask]
    if len(baseline_durations) < 5:
        raise ValueError(
            f"Only {len(baseline_durations)} baseline cycles for station {station} "
            f"in [{baseline_start}, {baseline_end}) -- too few to establish limits. "
            f"Widen the baseline window or check the station has data in this range."
        )

    limits = i_mr_limits(baseline_durations)

    monitor_times = times[monitor_mask]
    monitor_durations = durations[monitor_mask]
    flags = western_electric_flags(monitor_durations, limits)
    flags_with_time = [(float(monitor_times[i]), rule) for i, rule in flags]

    return times.tolist(), durations.tolist(), limits, flags_with_time
