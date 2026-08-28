import simpy

from .failures import is_night_shift, build_fault_cfg_for_station
from .variants import generate_build_sequence


class Part:
    __slots__ = ("part_id", "variant")

    def __init__(self, part_id, variant):
        self.part_id = part_id
        self.variant = variant


def resolve_tiers(cfg):
    """station_id -> 'instrumented' | 'partial' | 'manual', from the shops config."""
    tiers = {}
    for shop in cfg["shops"].values():
        for sid in shop["stations"]:
            tiers[sid] = shop["default_tier"]
        for sid_str, tier in shop.get("overrides", {}).items():
            tiers[int(sid_str)] = tier
    return tiers


class Station:
    """Four states: Running, Down, Blocked, Starved.

    Blocking/starving come for free from simpy.Store's own behaviour: a
    put() into a full Store doesn't complete until there's room (that's
    Blocked); a get() from an empty Store doesn't complete until a part
    arrives (that's Starved). We just log the transition before we yield.

    Simplification made deliberately for the sprint timeline: failures
    (Down) are checked between parts, not by interrupting mid-cycle. This
    is a standard simplification for a fast build and is stated here, and
    in the README, rather than hidden.
    """

    def __init__(self, env, station_id, cfg, rng, in_buf, out_buf, tier,
                 variant_multipliers, records):
        self.env = env
        self.id = station_id
        self.cfg = cfg
        self.rng = rng
        self.in_buf = in_buf
        self.out_buf = out_buf
        self.tier = tier
        self.variant_multipliers = variant_multipliers
        self.fault_cfg = build_fault_cfg_for_station(cfg, station_id)

        self.event_records = records["events"]
        self.state_records = records["states"]

        self.state = "Starved"
        self.state_since = 0.0
        self.busy_time = 0.0
        self.next_failure_time = self._sample_next_failure()

        self.action = env.process(self.run())

    def _sample_next_failure(self):
        mtbf = self.fault_cfg["mtbf_minutes"]
        return self.env.now + self.rng.expovariate(1.0 / mtbf)

    def _set_state(self, new_state):
        if new_state != self.state:
            self.state_records.append((self.id, self.state, self.state_since, self.env.now))
            self.state = new_state
            self.state_since = self.env.now

    def _base_processing_time(self, part):
        pt = self.cfg["processing_time"]
        takt_min = self.cfg["takt_seconds"] / 60.0
        frac = self.rng.triangular(pt["low"], pt["high"], pt["mode"])
        time = frac * takt_min

        mult = self.variant_multipliers.get(part.variant, {}).get(self.id, 1.0)
        time *= mult

        ew = self.fault_cfg.get("equipment_wear")
        if ew and self.env.now > ew["start_minute"]:
            drift = 1.0 + ew["slope"] * (self.env.now - ew["start_minute"])
            time *= drift

        cs = self.fault_cfg.get("constant_slowdown")
        if cs:
            time *= cs["multiplier"]

        base_sigma = 0.04
        sigma = base_sigma
        ov = self.fault_cfg.get("operator_variation")
        if ov and is_night_shift(self.env.now, ov["night_shift_hours"]):
            sigma = base_sigma * ov["night_shift_variance_multiplier"]
        noise = max(0.5, self.rng.normalvariate(1.0, sigma))
        time *= noise

        return max(0.1, time)

    def run(self):
        while True:
            # Check for failure BEFORE pulling the next part -- a station
            # can be down independent of whether work happens to be
            # queued for it. This also keeps t_start (captured right after
            # get() succeeds, below) an accurate record of the true moment
            # the part left the buffer, with no down-time-induced gap that
            # would otherwise inflate buffer-occupancy discovery.
            if self.env.now >= self.next_failure_time:
                self._set_state("Down")
                mttr = self.fault_cfg["mttr_minutes"]
                yield self.env.timeout(self.rng.expovariate(1.0 / mttr))
                self.next_failure_time = self._sample_next_failure()

            if len(self.in_buf.items) == 0:
                self._set_state("Starved")
            part = yield self.in_buf.get()
            self._set_state("Running")

            t_start = self.env.now
            proc_time = self._base_processing_time(part)
            yield self.env.timeout(proc_time)
            t_finish = self.env.now
            self.busy_time += proc_time

            self.event_records.append((part.part_id, self.id, t_start, t_finish, "OK", "NO"))

            if len(self.out_buf.items) >= self.out_buf.capacity:
                self._set_state("Blocked")
            yield self.out_buf.put(part)


def release_process(env, entry_buf, build_sequence):
    for row in build_sequence:
        delay = row["planned_release_minute"] - env.now
        if delay > 0:
            yield env.timeout(delay)
        yield entry_buf.put(Part(row["part_id"], row["variant"]))


def build_and_run(cfg, rng, variant_multipliers):
    env = simpy.Environment()
    records = {"events": [], "states": []}
    n = cfg["n_stations"]

    # buffers[0] = entry (uncapacitated), buffers[i] = after station i for
    # i in 1..n-1 (capacitated), buffers[n] = exit (uncapacitated)
    buffers = [simpy.Store(env) for _ in range(n + 1)]
    large = cfg["buffers"]["large_buffer"]
    lo, hi = cfg["buffers"]["default_range"]
    for i in range(1, n):
        cap = large["capacity"] if i == large["after_station"] else rng.randint(lo, hi)
        buffers[i] = simpy.Store(env, capacity=cap)

    # Ground-truth buffer capacities, captured for Phase 2's discovery accuracy
    # test. Purely additive -- doesn't consume any RNG draws or change any
    # existing output, so it can't affect Phase 1's already-verified numbers.
    cfg["_buffer_truth"] = {i: buffers[i].capacity for i in range(1, n)}

    tiers = resolve_tiers(cfg)
    stations = [
        Station(env, sid, cfg, rng, buffers[sid - 1], buffers[sid], tiers[sid],
                variant_multipliers, records)
        for sid in range(1, n + 1)
    ]

    build_seq = generate_build_sequence(cfg, rng)
    env.process(release_process(env, buffers[0], build_seq))

    env.run(until=cfg["simulation"]["run_minutes"])

    for st in stations:
        records["states"].append((st.id, st.state, st.state_since, env.now))

    # Ground truth, for Phase 2's discovery validation only -- twin/ never
    # reads this, it's purely so we can score how well discovery recovered
    # the real structure. Pure addition, doesn't touch anything above.
    records["ground_truth"] = {
        "topology_edges": [[i, i + 1] for i in range(1, n)],
        "buffer_capacity": {f"{i}->{i+1}": buffers[i].capacity for i in range(1, n)},
    }

    return records, stations, build_seq
