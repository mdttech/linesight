"""Acceptance test: Little's Law consistency check, against real data and
a deliberately broken case to confirm the flag actually fires."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plant.run import run_plant
from twin.forecast.physics_check import current_wip, recent_throughput, recent_flow_time, little_law_consistency


def main():
    records, stations, build_seq, cfg = run_plant("config/line_siteA.yaml", "plant_out_physics", seed=42)
    n = cfg["n_stations"]
    now, window = 1500.0, 120.0

    wip = current_wip(records["events"], 1, n, now)
    throughput = recent_throughput(records["events"], n, now, window)
    flow_time = recent_flow_time(records["events"], 1, n, now, window)
    result = little_law_consistency(wip, throughput, flow_time)

    print(f"WIP={wip}, throughput={throughput:.4f} parts/min, flow_time={flow_time:.2f} min")
    print(f"Predicted WIP: {result['predicted_wip']:.2f}  Observed: {wip}  Error: {result['error']*100:.1f}%")
    print(f"Consistent (within 15%): {result['consistent']} "
          f"({'PASS' if result['consistent'] else 'FAIL'} - expect a real steady-state system to pass)")

    broken = little_law_consistency(wip=wip, throughput=throughput * 0.4, flow_time=flow_time)
    print(f"\nDeliberately broken case: error={broken['error']*100:.1f}%, "
          f"flagged={not broken['consistent']} "
          f"({'PASS' if not broken['consistent'] else 'FAIL'} - the flag must fire here)")


if __name__ == "__main__":
    main()
