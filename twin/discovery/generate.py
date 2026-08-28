"""
Discovers a graph model of the line directly from an event log, following
Lugaresi & Matta (2021), Algorithm 2:

  1. Traces: group by part_id, ordered by ts_start -> the sequence of
     activities each part visited.
  2. Activity relations: an edge for every consecutive pair in every trace.
  3. Frequencies: how often each node and each edge occurs.
  4. Buffer capacities: for arc n->m, the maximum number of parts ever
     simultaneously "in the buffer" -- finished at n but not yet started
     at m. Computed here as a running max of a +1/-1 delta stream, which
     is the direct implementation of the paper's formula.

Processing time per node is stored as the raw array of observed durations
(an empirical distribution to resample from) rather than fitting a
parametric distribution to it -- this is deliberate, per the paper.
"""
import numpy as np
import networkx as nx


def build_traces(df):
    """part_id -> ordered list of (activity, ts_start, ts_finish)."""
    traces = {}
    for pid, g in df.groupby("part_id"):
        g = g.sort_values("ts_start")
        traces[pid] = list(zip(g["activity"], g["ts_start"], g["ts_finish"]))
    return traces


def _buffer_capacity(df, a, b):
    """Running max of (parts finished at a) - (parts started at b),
    i.e. the observed peak occupancy of the buffer between a and b."""
    finishes = df.loc[df["activity"] == a, "ts_finish"].to_numpy()
    starts = df.loc[df["activity"] == b, "ts_start"].to_numpy()

    events = np.concatenate([finishes, starts])
    deltas = np.concatenate([np.ones(len(finishes)), -np.ones(len(starts))])
    order = np.argsort(events, kind="stable")
    running = np.cumsum(deltas[order])
    return int(running.max()) if len(running) else 0


def generate_model(df):
    """df: the event log, columns part_id, activity, ts_start, ts_finish.
    ts_* can be numeric (minutes) or datetimes -- only relative order and
    differences matter here."""
    traces = build_traces(df)

    G = nx.DiGraph()
    edge_freq = {}
    for seq in traces.values():
        activities = [a for a, _, _ in seq]
        for a, b in zip(activities, activities[1:]):
            edge_freq[(a, b)] = edge_freq.get((a, b), 0) + 1

    node_freq = df["activity"].value_counts().to_dict()
    for node, freq in node_freq.items():
        sub = df.loc[df["activity"] == node]
        raw_durations = sub["ts_finish"] - sub["ts_start"]
        # works whether ts_* are plain floats (minutes) or parsed datetimes
        if hasattr(raw_durations, "dt"):
            durations = (raw_durations.dt.total_seconds() / 60.0).to_numpy()
        else:
            durations = raw_durations.to_numpy(dtype=float)
        G.add_node(node, freq=int(freq), durations=durations)

    for (a, b), freq in edge_freq.items():
        cap = _buffer_capacity(df, a, b)
        G.add_edge(a, b, freq=freq, capacity=cap)

    for node in G.nodes:
        out_edges = list(G.out_edges(node))
        total = sum(G.edges[e]["freq"] for e in out_edges)
        for e in out_edges:
            G.edges[e]["routing_prob"] = G.edges[e]["freq"] / total if total else 0.0

    return G


def sample_processing_time(G, node, rng):
    """Bootstrap-resample from the node's empirical distribution of
    observed durations -- 'empirical CDF, not a fitted distribution'."""
    durations = G.nodes[node]["durations"]
    return float(rng.choice(durations))
