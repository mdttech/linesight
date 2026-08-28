"""
The real plant is a simple 12-station linear chain with no branching, so
raw discovery already recovers the minimal true graph exactly -- there's
nothing for tune_model() to reduce, and target_size=12 makes it a correct
no-op. This test exercises the actual reduction mechanism in isolation, on
a small hand-built graph designed to need it, so the mechanism itself is
verified even though the real plant doesn't exercise it this sprint.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import networkx as nx
from twin.discovery.tune import tune_model


def build_toy_graph():
    """5 nodes in a line: 1->2->3->4->5. Node 3 is deliberately given a
    much lower frequency than its neighbours, so a frequency-driven
    reduction to 4 nodes should remove node 3 specifically."""
    G = nx.DiGraph()
    freqs = {1: 100, 2: 100, 3: 5, 4: 100, 5: 100}
    for n, f in freqs.items():
        G.add_node(n, freq=f)
    edges = [(1, 2, 10), (2, 3, 10), (3, 4, 10), (4, 5, 10)]
    for a, b, cap in edges:
        G.add_edge(a, b, capacity=cap, freq=freqs[a])
    return G


def main():
    G = build_toy_graph()
    print(f"Before tuning: {len(G.nodes)} nodes -- {sorted(G.nodes)}")

    tuned = tune_model(G, target_size=4)
    print(f"After tuning to target_size=4: {len(tuned.nodes)} nodes -- {sorted(tuned.nodes)}")

    removed_the_low_freq_node = 3 not in tuned.nodes
    hit_target_size = len(tuned.nodes) == 4
    print(f"\nNode 3 (lowest frequency) was merged away: "
          f"{'PASS' if removed_the_low_freq_node else 'FAIL'}")
    print(f"Reached the target size exactly: {'PASS' if hit_target_size else 'FAIL'}")

    # no-op check: target_size >= current size should change nothing
    G2 = build_toy_graph()
    noop = tune_model(G2, target_size=5)
    print(f"\nNo-op when target_size == current size: "
          f"{'PASS' if len(noop.nodes) == 5 else 'FAIL'}")


if __name__ == "__main__":
    main()
