"""
Trains the bottleneck-risk classifier. Generates training data from
several seeded 48-hour runs (labels are free -- see build_training_table's
docstring), splits by time within each run (not randomly, to avoid
leakage), trains XGBoost with class-imbalance weighting, and evaluates
against a trivial "highest buffer level" baseline using PR-AUC, since the
label here is a minority class too, same treatment as the defect problem.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve

from plant.config_loader import load_config
from plant.variants import build_variant_multipliers
from plant.line import build_and_run
from twin.ai.build_training_table import build_training_table, FEATURE_COLUMNS


def resolve_tiers(cfg):
    tiers = {}
    for shop in cfg["shops"].values():
        for sid in shop["stations"]:
            tiers[sid] = shop["default_tier"]
        for sid_str, t in shop.get("overrides", {}).items():
            tiers[int(sid_str)] = t
    return tiers


def generate_dataset(cfg_path, seeds, sample_interval=2.0, horizon=20.0):
    cfg = load_config(cfg_path)
    station_ids = list(range(1, cfg["n_stations"] + 1))
    tiers = resolve_tiers(cfg)
    run_minutes = cfg["simulation"]["run_minutes"]

    tables = []
    for seed in seeds:
        rng = random.Random(seed)
        variant_multipliers = build_variant_multipliers(cfg)
        records, stations, build_seq = build_and_run(cfg, rng, variant_multipliers)
        table = build_training_table(
            records, build_seq, station_ids, tiers, run_minutes,
            sample_interval=sample_interval, horizon=horizon, run_id=seed)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def time_split(table, train_frac=0.7):
    """Split by time *within each run*, not randomly -- a sample from
    late in a run never leaks into training data for that same run."""
    parts_train, parts_test = [], []
    for run_id, g in table.groupby("run_id"):
        cutoff = g["t"].max() * train_frac
        parts_train.append(g[g["t"] < cutoff])
        parts_test.append(g[g["t"] >= cutoff])
    return pd.concat(parts_train, ignore_index=True), pd.concat(parts_test, ignore_index=True)


def train_classifier(train_df):
    pos = (train_df["label"] == 1).sum()
    neg = (train_df["label"] == 0).sum()
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        scale_pos_weight=neg / max(pos, 1),
        eval_metric="aucpr",
    )
    model.fit(train_df[FEATURE_COLUMNS], train_df["label"])
    return model


def recall_at_precision(y_true, y_score, target_precision):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ok = precision[:-1] >= target_precision
    return float(recall[:-1][ok].max()) if ok.any() else 0.0


def evaluate(model, test_df):
    proba = model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
    y = test_df["label"].values

    model_pr_auc = average_precision_score(y, proba)
    baseline_score = test_df["buffer_in"].values  # "highest buffer level" baseline
    baseline_pr_auc = average_precision_score(y, baseline_score)

    r20 = recall_at_precision(y, proba, 0.20)
    r40 = recall_at_precision(y, proba, 0.40)

    return {
        "model_pr_auc": model_pr_auc,
        "baseline_pr_auc": baseline_pr_auc,
        "recall_at_20pct_precision": r20,
        "recall_at_40pct_precision": r40,
        "n_test": len(test_df),
        "positive_rate": y.mean(),
    }


def main():
    seeds = list(range(200, 208))
    print(f"Generating training data from {len(seeds)} seeded 48h runs...")
    table = generate_dataset("config/line_siteA.yaml", seeds)
    print(f"  {len(table)} total samples, positive rate {table['label'].mean():.4f}\n")

    train_df, test_df = time_split(table)
    print(f"Train: {len(train_df)} samples ({train_df['t'].max():.0f} min cutoff per run)")
    print(f"Test:  {len(test_df)} samples\n")

    model = train_classifier(train_df)
    results = evaluate(model, test_df)

    print("=== Evaluation on held-out, time-split data ===")
    print(f"  Model PR-AUC:    {results['model_pr_auc']:.4f}")
    print(f"  Baseline PR-AUC: {results['baseline_pr_auc']:.4f}  "
          f"('predict whichever station has the highest buffer_in')")
    print(f"  Positive rate:   {results['positive_rate']:.4f}")
    print(f"  Recall @ 20% precision: {results['recall_at_20pct_precision']:.3f}")
    print(f"  Recall @ 40% precision: {results['recall_at_40pct_precision']:.3f}")

    beats_baseline = results["model_pr_auc"] > results["baseline_pr_auc"]
    print(f"\n  Model beats baseline: {'PASS' if beats_baseline else 'FAIL'}")

    return model, table, train_df, test_df, results


if __name__ == "__main__":
    main()
