"""
The demo moment: reconfigure the plant (add a station), regenerate the
discovered model from the new log, and confirm it adapts with zero manual
edits to any discovery code -- only the config changed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
import pandas as pd
from plant.config_loader import load_config
from plant.run import run_plant
from twin.discovery.generate import generate_model


def main():
    print("=== BEFORE: original 12-station line ===")
    records, stations, build_seq, cfg = run_plant(
        "config/line_siteA_nofaults.yaml", "plant_out_reconfig_before", seed=11
    )
    df_before = pd.read_csv("plant_out_reconfig_before/event_log.csv",
                             parse_dates=["ts_start", "ts_finish"])
    G_before = generate_model(df_before)
    print(f"Discovered: {len(G_before.nodes)} nodes, {len(G_before.edges)} edges")
    print(f"  nodes: {sorted(G_before.nodes)}")

    # --- reconfigure: add a 13th station to general assembly ---
    cfg13 = load_config("config/line_siteA_nofaults.yaml")
    cfg13["n_stations"] = 13
    cfg13["shops"]["general_assembly"]["stations"].append(13)
    with open("config/line_siteA_13station.yaml", "w") as f:
        import yaml
        yaml.safe_dump(cfg13, f, sort_keys=False)

    print("\n=== Reconfigured: added station 13. Re-running the SAME plant code, ===")
    print("=== SAME discovery code -- only the config file changed. ===\n")

    print("=== AFTER: 13-station line ===")
    records2, stations2, build_seq2, cfg2 = run_plant(
        "config/line_siteA_13station.yaml", "plant_out_reconfig_after", seed=11
    )
    df_after = pd.read_csv("plant_out_reconfig_after/event_log.csv",
                            parse_dates=["ts_start", "ts_finish"])
    G_after = generate_model(df_after)
    print(f"Discovered: {len(G_after.nodes)} nodes, {len(G_after.edges)} edges")
    print(f"  nodes: {sorted(G_after.nodes)}")

    print()
    correct_before = sorted(G_before.nodes) == list(range(1, 13)) and len(G_before.edges) == 11
    correct_after = sorted(G_after.nodes) == list(range(1, 14)) and len(G_after.edges) == 12
    print(f"Before: recovered exactly 12 nodes / 11 edges: {'PASS' if correct_before else 'FAIL'}")
    print(f"After:  recovered exactly 13 nodes / 12 edges: {'PASS' if correct_after else 'FAIL'}")
    print(f"\nZero lines of discovery code touched between these two runs.")


if __name__ == "__main__":
    main()
