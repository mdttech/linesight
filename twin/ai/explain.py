"""
Turns the trained classifier's prediction into "candidate causes, ranked,
not certain" -- the same framing as the state-contribution diagnostics
from earlier phases, now backed by a trained model's own feature
attribution rather than only structural arithmetic. Shown together (both
signals corroborating or disagreeing) rather than either alone.
"""
import numpy as np
import shap

from twin.ai.build_training_table import FEATURE_COLUMNS

READABLE_NAMES = {
    "buffer_in": "input buffer filling up",
    "buffer_out": "output buffer congestion",
    "recent_cycle_time_mean": "recent cycle time trending up",
    "recent_cycle_time_std": "recent cycle time instability",
    "upcoming_loaded_share": "upcoming Loaded-variant vehicles",
    "upcoming_mid_share": "upcoming Mid-variant vehicles",
    "state_Running": "currently running",
    "state_Down": "currently down",
    "state_Blocked": "currently blocked",
    "state_Starved": "currently starved",
    "tier_instrumented": "fully instrumented station",
    "tier_partial": "partially instrumented station",
    "tier_manual": "manually-checked station (low visibility)",
}


def build_explainer(model):
    return shap.TreeExplainer(model)


def ranked_candidate_causes(explainer, row_df, top_k=3):
    """row_df: a single-row DataFrame with FEATURE_COLUMNS. Returns the
    top_k features driving THIS prediction, ranked by |SHAP value|, with
    relative weight -- framed as candidates, not a verdict, exactly like
    the state-contribution breakdown from earlier phases."""
    shap_values = explainer.shap_values(row_df[FEATURE_COLUMNS])
    values = np.asarray(shap_values)[0]
    total = np.abs(values).sum()
    if total == 0:
        return []

    idx = np.argsort(-np.abs(values))[:top_k]
    out = []
    for i in idx:
        feat = FEATURE_COLUMNS[i]
        out.append({
            "feature": feat,
            "label": READABLE_NAMES.get(feat, feat),
            "shap_value": float(values[i]),
            "weight": float(abs(values[i]) / total),
            "direction": "raises risk" if values[i] > 0 else "lowers risk",
        })
    return out


def format_candidate_causes(causes):
    if not causes:
        return "No dominant driver identified."
    lines = ["Candidate causes (ranked, not certain):"]
    for i, c in enumerate(causes, 1):
        lines.append(f"  {i}. {c['label']} ({c['direction']})   weight {c['weight']:.2f}")
    return "\n".join(lines)
