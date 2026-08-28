"""
The intervention mechanism for the prediction-vs-detection experiment.
Every `decision_interval` minutes, identify a target station -- by
detection (active period on the recent real past) or by prediction (roll
forward the discovered model over the known upcoming build sequence) --
and apply a temporary speed_boost there, representing "a supervisor gives
this station extra attention" in the simplest form that's actually
implementable this sprint. FIFO gets no controller at all (mode='fifo' is
handled by simply not attaching one, in the experiment runner).

This is a deliberately simple proxy for "acting on the identified
bottleneck," not the CONWIP/DBR dispatching-rule machinery in the
Ragazzini paper -- the point being tested (predicted beats detected) is
the same; the intervention mechanism is scoped down for the timeline.
"""
from twin.bottleneck.active_period import momentary_bottleneck
from twin.sync.state import snapshot
from twin.sync.rollforward import roll_forward


def controller_factory(mode, G_tuned, decision_interval, detection_window,
                        prediction_horizon, boost_factor, seed_base):
    """Returns a controller_factory-compatible callable for build_and_run()."""

    def _controller(env, stations_by_id, records, build_seq):
        station_ids = sorted(stations_by_id.keys())
        step = 0
        while True:
            yield env.timeout(decision_interval)
            now = env.now
            step += 1

            for s in stations_by_id.values():
                s.speed_boost = 1.0

            if mode == "detected":
                target, _ = momentary_bottleneck(
                    records["states"], station_ids,
                    max(0.0, now - detection_window), now)
            elif mode == "predicted":
                snap = snapshot(records, build_seq, station_ids, now)
                upcoming = [r for r in build_seq if r["planned_release_minute"] >= now][:80]
                sim_records = roll_forward(
                    G_tuned, station_ids, snap, upcoming, now,
                    prediction_horizon, seed=seed_base * 10000 + step)
                target, _ = momentary_bottleneck(
                    sim_records, station_ids, now, now + prediction_horizon)
            else:
                continue

            stations_by_id[target].speed_boost = boost_factor

    return _controller
