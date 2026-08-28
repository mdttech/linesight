"""
Phase 4 acceptance test: train the bottleneck-risk classifier across
several seeded runs, evaluate on held-out time-split data, confirm it
beats the trivial "highest buffer level" baseline on PR-AUC, and show a
real SHAP-explained example. This is the AI layer -- a genuinely trained
model, not just simulation and classical bottleneck detection.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from twin.ai.train import main as train_main
from twin.ai.explain import build_explainer, ranked_candidate_causes, format_candidate_causes
from twin.ai.build_training_table import FEATURE_COLUMNS


def main():
    model, table, train_df, test_df, results = train_main()

    print("\n=== Acceptance check ===")
    beats = results["model_pr_auc"] > results["baseline_pr_auc"]
    print(f"Model PR-AUC ({results['model_pr_auc']:.4f}) > "
          f"baseline PR-AUC ({results['baseline_pr_auc']:.4f}): {'PASS' if beats else 'FAIL'}")

    print("\n=== SHAP explanation on a real high-risk example ===")
    explainer = build_explainer(model)
    proba = model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
    test_df = test_df.copy()
    test_df["risk"] = proba
    example = test_df.sort_values("risk", ascending=False).iloc[[0]]

    print(f"Station {int(example['station'].values[0])} at t={example['t'].values[0]:.0f} min "
          f"(run seed {int(example['run_id'].values[0])})")
    print(f"Predicted risk: {example['risk'].values[0]:.3f}")
    print(f"Actually became bottleneck within 20 min: {bool(example['label'].values[0])}")
    print()
    causes = ranked_candidate_causes(explainer, example)
    print(format_candidate_causes(causes))


if __name__ == "__main__":
    main()
