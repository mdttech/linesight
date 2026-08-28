"""
Reconstructs the plant's current state at a point in time -- purely from
logs, never from plant internals. This is a deliberate constraint, not an
oversight: it's exactly what a real deployment would have available (an
MES event stream and a station-state stream), and keeping the twin bound
to it is what makes the plant/twin separation mean something.

Two things get reconstructed:

  station_states -- current state (Running/Down/Blocked/Starved) and how
  long each station has been in it, read directly from state_log rows
  whose interval contains `now`.

  buffer_wip -- which parts are currently sitting in each inter-station
  buffer, with their variant. Computed the same way Phase 2's buffer
  capacity discovery counts occupancy: parts finished upstream but not
  yet started downstream, evaluated at a single instant instead of as a
  running max over the whole log. Variant is looked up from the build
  sequence, since a real MES tracks vehicle configuration alongside
  position -- this isn't privileged simulator access.
"""


def current_station_states(state_records, stations, now):
    """{station_id: (state, minutes_in_state)}"""
    out = {}
    for s, state, start, end in state_records:
        if start <= now < end or (end == now and start <= now):
            out[s] = (state, now - start)
    latest = {}
    for s, state, start, end in state_records:
        if s not in latest or start > latest[s][1]:
            latest[s] = (state, start, end)
    for s in stations:
        if s not in out and s in latest:
            state, start, end = latest[s]
            out[s] = (state, max(0.0, now - start))
    return out


def current_buffer_wip(events, build_seq, stations, now):
    """{(a, b): [variant, variant, ...]} -- parts finished at a but not
    yet started at b, oldest first, by wall-clock finish order."""
    variant_of = {row["part_id"]: row["variant"] for row in build_seq}

    finished_at = {s: [] for s in stations}
    started_at = {s: set() for s in stations}

    for part_id, station, t_start, t_finish, result, scrap in events:
        if t_finish <= now:
            finished_at[station].append((t_finish, part_id))
        if t_start <= now:
            started_at[station].add(part_id)

    wip = {}
    for i in range(len(stations) - 1):
        a, b = stations[i], stations[i + 1]
        waiting = [
            pid for t_fin, pid in sorted(finished_at[a])
            if pid not in started_at[b]
        ]
        wip[(a, b)] = [variant_of.get(pid, "Base") for pid in waiting]
    return wip


def most_recent_variant(events, build_seq, station, now):
    """The variant of whichever part most recently finished at this
    station, as of `now`. Used as a reasonable stand-in for 'what a
    Running/Down/Blocked station is currently holding' when its input
    buffer is empty -- a real, log-derived inference, not a guess pulled
    from nowhere: it's simply the most recent thing we've actually
    observed that station work on."""
    variant_of = {row["part_id"]: row["variant"] for row in build_seq}
    candidates = [
        (t_finish, part_id) for part_id, st, t_start, t_finish, _, _ in events
        if st == station and t_finish <= now
    ]
    if not candidates:
        return "Base"
    _, part_id = max(candidates)
    return variant_of.get(part_id, "Base")


def snapshot(records, build_seq, stations, now):
    return {
        "station_states": current_station_states(records["states"], stations, now),
        "buffer_wip": current_buffer_wip(records["events"], build_seq, stations, now),
        "recent_variant": {
            s: most_recent_variant(records["events"], build_seq, s, now) for s in stations
        },
    }
