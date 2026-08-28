def is_night_shift(now_minutes, night_hours):
    """night_hours = [start_hour, end_hour], e.g. [22, 6] meaning 22:00-06:00.
    Handles the wrap past midnight."""
    hour = int((now_minutes // 60) % 24)
    start, end = night_hours
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


def build_fault_cfg_for_station(cfg, station_id):
    """Pulls out only the fault config relevant to this specific station,
    so Station doesn't need to know about the global config shape."""
    faults = cfg.get("faults", {})
    out = {
        "mtbf_minutes": cfg["reliability"]["mtbf_minutes"],
        "mttr_minutes": cfg["reliability"]["mttr_minutes"],
    }

    ew = faults.get("equipment_wear")
    if ew and ew["station"] == station_id:
        out["equipment_wear"] = ew

    ov = faults.get("operator_variation")
    if ov and station_id in ov["stations"]:
        out["operator_variation"] = ov

    # test-only lever, not one of the two "real" fault modes -- used by the
    # acceptance test to force one station slower with everything else off.
    cs = faults.get("constant_slowdown")
    if cs and cs["station"] == station_id:
        out["constant_slowdown"] = cs

    return out
