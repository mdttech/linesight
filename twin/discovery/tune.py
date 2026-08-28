"""
Model tuning, following Lugaresi & Matta (2021), Algorithm 1: a local
search that reduces the graph toward a target node count while maximising
an adequacy score. We implement three of the five scoring terms from the
paper (buffer sizes, split/merge points, activity frequency), equally
weighted, and the "Frequency" neighbour-generation rule (aggregate the
lowest-frequency node into its highest-capacity neighbour first).

Honest scope note: on a simple linear-chain plant (no branching, no
merging), raw discovery already recovers the minimal true graph exactly,
so tuning with target_size == true size is correctly a no-op here -- there
is nothing to prune. The reduction mechanism itself is still implemented
and tested in isolation (see tests/test_tune_unit.py) against a small
synthetic graph built to actually need reduction, since the real plant's
topology doesn't exercise it.
"""
import networkx as nx


def _buffer_term(G, G_original):
    total_original = sum(d["capacity"] for _, _, d in G_original.edges(data=True))
    total_remaining = sum(d["capacity"] for _, _, d in G.edges(data=True))
    return total_remaining / total_original if total_original else 1.0


def _split_merge_term(G, G_original):
    split_merge_original = [
        n for n in G_original.nodes
        if G_original.out_degree(n) > 1 or G_original.in_degree(n) > 1
    ]
    if not split_merge_original:
        return 1.0
    preserved = sum(1 for n in split_merge_original if n in G.nodes)
    return preserved / len(split_merge_original)


def _frequency_term(G, G_original):
    total_original = sum(d["freq"] for _, d in G_original.nodes(data=True))
    total_remaining = sum(d["freq"] for _, d in G.nodes(data=True))
    return total_remaining / total_original if total_original else 1.0


def adequacy_score(G, G_original):
    """Phi(Omega) = mean of three ratios, each in [0, 1]. Equal weights."""
    return (
        _buffer_term(G, G_original)
        + _split_merge_term(G, G_original)
        + _frequency_term(G, G_original)
    ) / 3.0


def _merge_node(G, node):
    """Merge `node` into its highest-capacity neighbour (preferring the
    connection that carried the most buffered material, on the theory
    that it's the connection most worth preserving)."""
    G = G.copy()
    in_edges = list(G.in_edges(node, data=True))
    out_edges = list(G.out_edges(node, data=True))
    neighbours = [(u, d) for u, _, d in in_edges] + [(v, d) for _, v, d in out_edges]
    if not neighbours:
        G.remove_node(node)
        return G

    target, _ = max(neighbours, key=lambda nd: nd[1].get("capacity", 0))

    # fold this node's frequency into the surviving neighbour
    G.nodes[target]["freq"] = G.nodes[target].get("freq", 0) + G.nodes[node].get("freq", 0)

    # reconnect: anything that pointed into `node` now points into `target`;
    # anything `node` pointed to, `target` now points to (skip self-loops)
    for u, _, d in in_edges:
        if u != target:
            G.add_edge(u, target, **d)
    for _, v, d in out_edges:
        if v != target:
            G.add_edge(target, v, **d)

    G.remove_node(node)
    return G


def tune_model(G, target_size, max_iterations=200):
    """Reduce G toward target_size nodes, picking at each step whichever
    single-node removal keeps the adequacy score highest. No-ops cleanly
    if G is already at or below target_size -- which is the case for this
    sprint's plant, since its linear-chain topology has no spaghetti to
    prune in the first place."""
    G_original = G.copy()
    current = G.copy()

    iterations = 0
    while len(current.nodes) > target_size and iterations < max_iterations:
        # Frequency rule: consider removing the lowest-frequency nodes first
        candidates = sorted(current.nodes, key=lambda n: current.nodes[n]["freq"])[:3]

        best_graph, best_score = None, -1.0
        for node in candidates:
            candidate_graph = _merge_node(current, node)
            score = adequacy_score(candidate_graph, G_original)
            if score > best_score:
                best_graph, best_score = candidate_graph, score

        current = best_graph
        iterations += 1

    return current
