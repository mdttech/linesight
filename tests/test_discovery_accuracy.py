"""
Acceptance test: node/arc precision-recall and buffer capacity MAE vs. the
plant's true structure. Ground truth comes from the plant's own topology
(a linear chain 1->2->...->n) and from cfg["_buffer_truth"], captured
directly by the simulator -- not reconstructed separately, so there's no
risk of the ground-truth computation drifting from what the plant actually
built.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from plant.run import run_plant
from twin.discovery.generate import generate_model


def true_structure(cfg):
    n = cfg["n_stations"]
    nodes = set(range(1, n + 1))
    edges = set((i, i + 1) for i in range(1, n))
    buffer_truth = cfg["_buffer_truth"]  # {station_after: capacity}
    edge_capacity_truth = {(i, i + 1): buffer_truth[i] for i in range(1, n)}
    return nodes, edges, edge_capacity_truth


def precision_recall(discovered, true):
    discovered, true = set(discovered), set(true)
    tp = len(discovered & true)
    precision = tp / len(discovered) if discovered else 0.0
    recall = tp / len(true) if true else 0.0
    return precision, recall


def main():
    # A no-faults run gives clean, stationary data -- the right setting to
    # validate the discovery mechanism itself, separate from validating
    # that faults are visible in the data (that's what test_faults.py is for).
    records, stations, build_seq, cfg = run_plant(
        "config/line_siteA_nofaults.yaml", "plant_out_discovery", seed=99
    )

    df = pd.read_csv("plant_out_discovery/event_log.csv", parse_dates=["ts_start", "ts_finish"])
    G = generate_model(df)

    true_nodes, true_edges, true_caps = true_structure(cfg)

    node_p, node_r = precision_recall(G.nodes, true_nodes)
    edge_p, edge_r = precision_recall(G.edges, true_edges)

    print(f"Node precision: {node_p:.3f}  recall: {node_r:.3f}")
    print(f"Arc  precision: {edge_p:.3f}  recall: {edge_r:.3f}\n")

    print("--- Buffer capacity: discovered vs. true ---")
    errors = []
    for e in sorted(true_edges):
        true_cap = true_caps[e]
        disc_cap = G.edges[e]["capacity"] if e in G.edges else None
        if disc_cap is None:
            print(f"  {e}: MISSING from discovered graph")
            continue
        err = abs(disc_cap - true_cap)
        errors.append(err)
        flag = "  <-- the large buffer" if true_cap > 20 else ""
        print(f"  {e}: true={true_cap:4d}  discovered={disc_cap:4d}  |err|={err:3d}{flag}")

    mae = sum(errors) / len(errors)
    print(f"\nBuffer capacity MAE: {mae:.2f}")


if __name__ == "__main__":
    main()
