"""
Converts a discovered (and possibly tuned) graph directly into a runnable
SimPy model: each node becomes a station whose processing time is
bootstrap-resampled from its observed empirical distribution, each edge
becomes a simpy.Store with the discovered capacity. This is the twin
regenerating a runnable model with no manual edit -- just the event log in,
a running simulator out.
"""
import simpy

from .generate import sample_processing_time


class GeneratedStation:
    def __init__(self, env, node_id, G, rng, in_buf, out_buf, completions):
        self.env = env
        self.node_id = node_id
        self.G = G
        self.rng = rng
        self.in_buf = in_buf
        self.out_buf = out_buf
        self.completions = completions
        self.busy_time = 0.0
        self.action = env.process(self.run())

    def run(self):
        while True:
            part = yield self.in_buf.get()
            proc_time = sample_processing_time(self.G, self.node_id, self.rng)
            yield self.env.timeout(proc_time)
            self.busy_time += proc_time
            if self.out_buf is not None:
                yield self.out_buf.put(part)
            else:
                self.completions.append(self.env.now)


def run_generated_model(G, run_minutes, release_interval, rng, seed_parts=None):
    """Runs the generated model standalone: a simple release process feeds
    parts into the first node at a fixed interval, parts flow through the
    discovered topology, and we record completion times at terminal nodes
    (out-degree 0) to compute throughput."""
    env = simpy.Environment()

    order = list(G.nodes)
    entry_nodes = [n for n in order if G.in_degree(n) == 0]
    terminal_nodes = [n for n in order if G.out_degree(n) == 0]

    buffers = {}
    for u, v, data in G.edges(data=True):
        buffers[(u, v)] = simpy.Store(env, capacity=max(1, data["capacity"]))

    entry_buf = {n: simpy.Store(env) for n in entry_nodes}
    completions = []

    def out_buf_for(node):
        out_edges = list(G.out_edges(node))
        return buffers[out_edges[0]] if out_edges else None

    def in_buf_for(node):
        in_edges = list(G.in_edges(node))
        if in_edges:
            return buffers[in_edges[0]]
        return entry_buf[node]

    stations = [
        GeneratedStation(env, n, G, rng, in_buf_for(n), out_buf_for(n), completions)
        for n in order
    ]

    def release(env, node):
        t = 0
        while True:
            yield env.timeout(release_interval)
            t += release_interval
            yield entry_buf[node].put(object())

    for n in entry_nodes:
        env.process(release(env, n))

    env.run(until=run_minutes)
    return completions, stations
