import csv
from datetime import timedelta


def build_variant_multipliers(cfg):
    """{'Loaded': {4: 1.40, 7: 1.35, 9: 1.30}, ...} -- station keys as ints."""
    result = {}
    for name, spec in cfg["variants"].items():
        result[name] = {int(k): v for k, v in spec.get("multipliers", {}).items()}
    return result


def generate_build_sequence(cfg, rng):
    """The scheduled future -- known in advance, which is the whole basis
    of predicting rather than just detecting."""
    variants = cfg["variants"]
    names = list(variants.keys())
    shares = [variants[n]["share"] for n in names]
    takt_min = cfg["takt_seconds"] / 60.0
    n_parts = int(cfg["simulation"]["run_minutes"] / takt_min) + 5

    seq = []
    for i in range(n_parts):
        variant = rng.choices(names, weights=shares, k=1)[0]
        seq.append({
            "part_id": i + 1,
            "variant": variant,
            "planned_release_minute": i * takt_min,
        })
    return seq


def write_build_sequence_csv(seq, path, anchor_dt):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["part_id", "variant", "planned_release"])
        for row in seq:
            ts = anchor_dt + timedelta(minutes=row["planned_release_minute"])
            w.writerow([row["part_id"], row["variant"], ts.strftime("%Y-%m-%d %H:%M:%S")])
