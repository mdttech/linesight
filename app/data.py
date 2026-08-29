"""
Everything the UI needs to display, computed here and fully testable
without ever starting Streamlit. app.py imports this and only handles
rendering -- deliberately, since a Streamlit app can't be smoke-tested
the same way every other phase's CLI scripts have been, but this module
can be, exactly like everything else in this project.

This is a REPLAY demo, not a live one: the plant is simulated once
(cached), and a time slider moves through that fixed run rather than
simulating in real time as the app runs. Disclosed here and in the app
itself -- a documented, deliberate simplification (see the Complete
Guide's risk register), not something to discover later.
"""
import random

import numpy as np
import pandas as pd

from plant.config_loader import load_config
from plant.variants import build_variant_multipliers
from plant.line import build_and_run
from twin.discovery.generate import generate_model, sample_processing_time_for_variant
from twin.discovery.tune import tune_model
from twin.sync.state import snapshot
from twin.sync.rollforward import roll_forward
from twin.bottleneck.active_period import momentary_bottleneck
from twin.ai.build_training_table import build_training_table, FEATURE_COLUMNS
from twin.ai.train import time_split, train_classifier
from twin.ai.explain import build_explainer, ranked_candidate_causes
from twin.ledger import PredictionLedger
from twin.forecast.physics_check import current_wip, recent_throughput, recent_flow_time, little_law_consistency

CONFIG_PATH = "config/line_siteA.yaml"
DEMO_SEED = 42
PREDICTION_HORIZON = 25.0
DECISION_INTERVAL = 60.0


def resolve_tiers(cfg):
    tiers = {}
    for shop in cfg["shops"].values():
        for sid in shop["stations"]:
            tiers[sid] = shop["default_tier"]
        for sid_str, t in shop.get("overrides", {}).items():
            tiers[int(sid_str)] = t
    return tiers


def build_demo_state():
    """The expensive one-time setup: run the plant, discover + tune the
    model, train the classifier, pre-compute a full ledger by replaying
    predictions across the run. Call this once (app.py wraps it in
    st.cache_resource) -- everything after is fast slider interaction."""
    cfg = load_config(CONFIG_PATH)
    station_ids = list(range(1, cfg["n_stations"] + 1))
    tiers = resolve_tiers(cfg)
    run_minutes = cfg["simulation"]["run_minutes"]

    rng = random.Random(DEMO_SEED)
    variant_multipliers = build_variant_multipliers(cfg)
    records, stations, build_seq = build_and_run(cfg, rng, variant_multipliers)

    df = pd.DataFrame(records["events"],
                       columns=["part_id", "activity", "ts_start", "ts_finish", "result", "scrap"])
    variant_map = {row["part_id"]: row["variant"] for row in build_seq}
    G = generate_model(df, variant_map=variant_map)
    G_tuned = tune_model(G, target_size=len(G.nodes))

    # classifier: trained on this same run plus a couple more seeds, same
    # approach as Phase 4, just fewer seeds here since the UI only needs
    # a working model, not a from-scratch PR-AUC benchmark
    tables = [build_training_table(records, build_seq, station_ids, tiers, run_minutes, run_id=DEMO_SEED)]
    for extra_seed in (DEMO_SEED + 1, DEMO_SEED + 2):
        r2 = random.Random(extra_seed)
        recs2, _, seq2 = build_and_run(cfg, r2, build_variant_multipliers(cfg))
        tables.append(build_training_table(recs2, seq2, station_ids, tiers, run_minutes, run_id=extra_seed))
    table = pd.concat(tables, ignore_index=True)
    train_df, _ = time_split(table, train_frac=0.85)
    model = train_classifier(train_df)
    explainer = build_explainer(model)

    # pre-compute the full replay ledger: at every decision point in the
    # run, log the predicted bottleneck, resolve it once we know what
    # actually happened at that horizon
    ledger = PredictionLedger()
    decision_points = np.arange(DECISION_INTERVAL, run_minutes - PREDICTION_HORIZON, DECISION_INTERVAL)
    pred_log = []  # (decision_time, pred_id, predicted_station)
    for t in decision_points:
        snap = snapshot(records, build_seq, station_ids, t)
        upcoming = [r for r in build_seq if r["planned_release_minute"] >= t][:80]
        sim_records = roll_forward(G_tuned, station_ids, snap, upcoming, t, PREDICTION_HORIZON, seed=int(t))
        predicted_station, _ = momentary_bottleneck(sim_records, station_ids, t, t + PREDICTION_HORIZON)
        pid = ledger.log(predicted_station, "bottleneck", t + PREDICTION_HORIZON,
                          confidence=0.75, made_at=t)
        pred_log.append((t, pid, predicted_station))

    for t, pid, predicted_station in pred_log:
        actual_station, _ = momentary_bottleneck(records["states"], station_ids, t, t + PREDICTION_HORIZON)
        ledger.resolve(pid, actual_station == predicted_station)

    return {
        "cfg": cfg, "station_ids": station_ids, "tiers": tiers,
        "records": records, "build_seq": build_seq, "run_minutes": run_minutes,
        "G_tuned": G_tuned, "model": model, "explainer": explainer,
        "ledger": ledger, "variant_map": variant_map,
    }


def current_view(state, now):
    """Everything the Floor tab needs at one instant: station states,
    buffer levels, the live prediction, the classifier's take, and the
    ledger's track record as of this point in the replay."""
    records, build_seq = state["records"], state["build_seq"]
    station_ids, cfg = state["station_ids"], state["cfg"]

    snap = snapshot(records, build_seq, station_ids, now)
    upcoming = [r for r in build_seq if r["planned_release_minute"] >= now][:80]
    sim_records = roll_forward(state["G_tuned"], station_ids, snap, upcoming, now,
                                PREDICTION_HORIZON, seed=int(now) + 1)
    predicted_station, active_min = momentary_bottleneck(sim_records, station_ids, now, now + PREDICTION_HORIZON)

    # the classifier's independent read on the predicted station right now
    tiers = state["tiers"]
    row = _feature_row_at(records, build_seq, predicted_station, now, tiers)
    proba = float(state["model"].predict_proba(row[FEATURE_COLUMNS])[:, 1][0])
    causes = ranked_candidate_causes(state["explainer"], row, top_k=3)

    wip = current_wip(records["events"], station_ids[0], station_ids[-1], now)
    throughput = recent_throughput(records["events"], station_ids[-1], now, window=120.0)
    flow_time = recent_flow_time(records["events"], station_ids[0], station_ids[-1], now, window=120.0)
    physics = little_law_consistency(wip, throughput, flow_time) if wip > 0 and throughput > 0 else None

    return {
        "station_states": snap["station_states"],
        "buffer_wip": snap["buffer_wip"],
        "predicted_station": predicted_station,
        "predicted_active_minutes": active_min,
        "classifier_probability": proba,
        "candidate_causes": causes,
        "physics_check": physics,
        "ledger_summary": _ledger_summary_as_of(state["ledger"], now),
    }


def _feature_row_at(records, build_seq, station, now, tiers):
    """One-row feature table for `station` at `now`, same columns
    build_training_table produces, for feeding to the trained classifier
    and its SHAP explainer."""
    from twin.ai.build_training_table import (
        _buffer_step_function, _lookup_step, _state_step_function, _lookup_state,
        _recent_cycle_stats, _upcoming_shares, STATES, TIERS,
    )
    station_ids = sorted(tiers.keys())
    idx = station_ids.index(station)

    buf_in = 0.0
    if idx > 0:
        t, v = _buffer_step_function(records["events"], station_ids[idx - 1], station, now)
        buf_in = float(_lookup_step(t, v, np.array([now]))[0])
    buf_out = 0.0
    if idx < len(station_ids) - 1:
        t, v = _buffer_step_function(records["events"], station, station_ids[idx + 1], now)
        buf_out = float(_lookup_step(t, v, np.array([now]))[0])

    state_t, state_v = _state_step_function(records["states"], station)
    cur_state = _lookup_state(state_t, state_v, np.array([now]))[0]
    cyc_mean, cyc_std = _recent_cycle_stats(records["events"], station, np.array([now]))
    up_loaded, up_mid = _upcoming_shares(build_seq, np.array([now]))

    row = {
        "buffer_in": buf_in, "buffer_out": buf_out,
        "recent_cycle_time_mean": float(cyc_mean[0]), "recent_cycle_time_std": float(cyc_std[0]),
        "upcoming_loaded_share": float(up_loaded[0]), "upcoming_mid_share": float(up_mid[0]),
    }
    for s in STATES:
        row[f"state_{s}"] = 1 if cur_state == s else 0
    for tr in TIERS:
        row[f"tier_{tr}"] = 1 if tiers[station] == tr else 0
    return pd.DataFrame([row])


def _ledger_summary_as_of(ledger, now):
    """Only counts predictions whose full horizon has already passed at
    `now` -- filtering on made_at instead of predicted_for was a real bug
    caught while testing this: it let the replay claim to 'know' the
    outcome of a prediction whose 25-minute window hadn't finished yet,
    which would have been a findable integrity gap in a live demo."""
    visible = [p for p in ledger.all_predictions() if p["predicted_for"] <= now]
    confirmed = sum(1 for p in visible if p["status"] == "confirmed")
    false_alarms = sum(1 for p in visible if p["status"] == "false_alarm")
    pending = sum(1 for p in visible if p["status"] == "pending")
    return {
        "confirmed": confirmed, "false_alarms": false_alarms, "pending": pending,
        "text": f"last {confirmed + false_alarms} predictions: {confirmed} confirmed, "
                f"{false_alarms} false alarm{'s' if false_alarms != 1 else ''}",
    }


def business_case_numbers():
    """The real, computed business case -- not hardcoded placeholders.
    See docs/business_case_notes.md for the full derivation and the
    sensitivity check on the throughput-value assumption."""
    assumptions = {
        "cost_per_late_defect_usd": 1200, "cost_per_early_defect_usd": 80,
        "value_per_throughput_hour_usd": 4500, "camera_hardware_cost_usd": 450,
        "camera_install_labor_usd": 300, "annual_operating_hours": 6000,
    }
    annual_vehicles = int(assumptions["annual_operating_hours"] * 3600 / 60)
    defects_per_year = annual_vehicles * 0.0058
    caught_early = defects_per_year * 0.157
    savings_defects = caught_early * (assumptions["cost_per_late_defect_usd"] - assumptions["cost_per_early_defect_usd"])
    throughput_hours = assumptions["annual_operating_hours"] * 0.0728
    savings_throughput = throughput_hours * assumptions["value_per_throughput_hour_usd"]
    total_savings = savings_defects + savings_throughput
    one_time_cost = 5 * (assumptions["camera_hardware_cost_usd"] + assumptions["camera_install_labor_usd"])
    payback_months = one_time_cost / (total_savings / 12)
    return {
        "annual_vehicles": annual_vehicles, "defects_caught_early": caught_early,
        "savings_defects": savings_defects, "savings_throughput": savings_throughput,
        "total_annual_savings": total_savings, "one_time_cost": one_time_cost,
        "payback_months": payback_months, "assumptions": assumptions,
    }
