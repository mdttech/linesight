import argparse
import os
import random
from datetime import datetime

from .config_loader import load_config
from .variants import build_variant_multipliers, write_build_sequence_csv
from .line import build_and_run
from .emit import write_event_log, write_state_log


def run_plant(config_path, out_dir, seed=None):
    cfg = load_config(config_path)
    if seed is not None:
        cfg["seed"] = seed

    rng = random.Random(cfg["seed"])
    variant_multipliers = build_variant_multipliers(cfg)

    os.makedirs(out_dir, exist_ok=True)
    anchor_dt = datetime(2026, 3, 18, 6, 0, 0)

    records, stations, build_seq = build_and_run(cfg, rng, variant_multipliers)

    write_event_log(records["events"], os.path.join(out_dir, "event_log.csv"), anchor_dt)
    write_state_log(records["states"], os.path.join(out_dir, "state_log.csv"), anchor_dt)
    write_build_sequence_csv(build_seq, os.path.join(out_dir, "build_sequence.csv"), anchor_dt)

    return records, stations, build_seq, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/line_siteA.yaml")
    parser.add_argument("--out", default="plant_out")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    records, stations, build_seq, cfg = run_plant(args.config, args.out, args.seed)
    print(f"Wrote {len(records['events'])} events, {len(records['states'])} state "
          f"transitions across {len(stations)} stations to {args.out}/")


if __name__ == "__main__":
    main()
