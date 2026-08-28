"""
The prediction mechanism. Synchronise the discovered/tuned model to the
plant's current state (twin.sync.state.snapshot), then simulate it forward
over the vehicle build sequence for a short horizon -- which is scheduled
and already known, not guessed. Running active_period on the *simulated*
future window this produces is the prediction; running it on real past
data (twin.bottleneck.active_period, directly) is the detection.

Simplification made deliberately for the sprint timeline: a station that
is mid-cycle (Running or Down) at synchronisation time is restarted with
a freshly-sampled duration in that state, rather than resuming with an
exact remaining-time countdown. This slightly overstates near-term
availability for a station that's about to finish its current cycle, but
correctly captures *which* stations are constrained and roughly by how
much -- the dominant signal for a short-horizon prediction. Documented
here rather than fixed under this timeline; state.py's snapshot already
does the harder, more important part correctly (WIP position and variant,
reconstructed purely from logs).
"""
import random
import simpy

from twin.discovery.generate import sample_processing_time_for_variant


class _RollForwardStation:
    def __init__(self, env, station_id, G, rng, in_buf, out_buf, initial_state,
                 records, phantom_variant):
        self.env = env
        self.station_id = station_id
        self.G = G
        self.rng = rng
        self.in_buf = in_buf
        self.out_buf = out_buf
        self.records = records
        self.state = "Starved"
        self.state_since = env.now
        self.action = env.process(self.run(initial_state, phantom_variant))

    def _set_state(self, new_state):
        if new_state != self.state:
            self.records.append((self.station_id, self.state, self.state_since, self.env.now))
            self.state, self.state_since = new_state, self.env.now

    def run(self, initial_state, phantom_variant):
        if initial_state == "Down":
            self._set_state("Down")
            yield self.env.timeout(self.rng.expovariate(1.0 / 5.0))
            self._set_state("Running")
        elif initial_state in ("Running", "Blocked") and len(self.in_buf.items) == 0:
            # snapshotted as mid-cycle but its buffer is empty -- it's
            # already holding a part that isn't reflected in any buffer.
            # Give it that phantom part now rather than making it wait
            # for a delivery it doesn't actually need.
            self._set_state("Running")
            proc_time = sample_processing_time_for_variant(
                self.G, self.station_id, phantom_variant, self.rng)
            yield self.env.timeout(proc_time)
            if self.out_buf is not None:
                if len(self.out_buf.items) >= self.out_buf.capacity:
                    self._set_state("Blocked")
                yield self.out_buf.put(phantom_variant)

        while True:
            if len(self.in_buf.items) == 0:
                self._set_state("Starved")
            variant = yield self.in_buf.get()
            self._set_state("Running")

            proc_time = sample_processing_time_for_variant(self.G, self.station_id, variant, self.rng)
            yield self.env.timeout(proc_time)

            if self.out_buf is not None:
                if len(self.out_buf.items) >= self.out_buf.capacity:
                    self._set_state("Blocked")
                yield self.out_buf.put(variant)


def roll_forward(G, stations, snapshot_data, upcoming_build_seq, now, horizon_minutes, seed):
    """Simulates `horizon_minutes` forward from `now`, initialised from
    `snapshot_data` (twin.sync.state.snapshot output) and fed the known
    upcoming build sequence. Returns simulated state records for the
    window [now, now+horizon] -- feed straight into
    twin.bottleneck.active_period.momentary_bottleneck() for the
    prediction."""
    rng = random.Random(seed)
    env = simpy.Environment(initial_time=now)
    records = []

    buffers = {}
    for i in range(len(stations) - 1):
        a, b = stations[i], stations[i + 1]
        cap = G.edges[(a, b)]["capacity"] if G.has_edge(a, b) else 10
        buf = simpy.Store(env, capacity=cap)
        for variant in snapshot_data["buffer_wip"].get((a, b), []):
            buf.put(variant)
        buffers[(a, b)] = buf

    station_objs = []

    entry_buf = simpy.Store(env)
    exit_store = simpy.Store(env)

    def buf_before(sid):
        idx = stations.index(sid)
        return entry_buf if idx == 0 else buffers[(stations[idx - 1], sid)]

    def buf_after(sid):
        idx = stations.index(sid)
        return None if idx == len(stations) - 1 else buffers[(sid, stations[idx + 1])]

    for sid in stations:
        init_state = snapshot_data["station_states"].get(sid, ("Starved", 0.0))[0]
        phantom_variant = snapshot_data.get("recent_variant", {}).get(sid, "Base")
        out_buf = buf_after(sid)
        # The entry station's "buffer" is the scheduled release process
        # itself, not a pool of already-finished upstream work -- it
        # doesn't need a phantom part, the next scheduled arrival already
        # represents what it will pick up next. Giving it one anyway
        # double-counts a unit of work that's also arriving on schedule.
        is_entry = (sid == stations[0])
        station_objs.append(_RollForwardStation(
            env, sid, G, rng, buf_before(sid), out_buf or exit_store,
            "Starved" if is_entry else init_state, records, phantom_variant))

    def release(env, entry_buf, upcoming):
        for row in upcoming:
            delay = row["planned_release_minute"] - env.now
            if delay > 0:
                yield env.timeout(delay)
            yield entry_buf.put(row["variant"])
    env.process(release(env, entry_buf, upcoming_build_seq))

    env.run(until=now + horizon_minutes)

    # flush each station's still-open interval, exactly as build_and_run()
    # does -- without this, a station mid-cycle when the horizon ends
    # (e.g. a long draw that outlasts the window) silently loses its
    # entire final period from the record, which is a real bug, not a
    # rare edge case: any station busy near the end of the window hits it.
    end_time = now + horizon_minutes
    for st in station_objs:
        records.append((st.station_id, st.state, st.state_since, end_time))

    return records
