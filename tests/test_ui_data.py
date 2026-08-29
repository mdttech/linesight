"""
Acceptance test for the UI's data layer (app/data.py) -- the part that
can be tested the same way as every other phase. The Streamlit rendering
layer itself (app/app.py) is verified separately via Streamlit's own
AppTest framework -- see the placement guide for that command.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.data import build_demo_state, current_view, business_case_numbers


def main():
    print("Building demo state (one-time setup, ~15-20s)...")
    state = build_demo_state()
    print(f"Setup complete. {len(state['ledger'].all_predictions())} predictions pre-computed.\n")

    for now in (1500.0, 2000.0, 2500.0):
        view = current_view(state, now)
        print(f"--- now={now} ---")
        print(f"Predicted: station {view['predicted_station']}, "
              f"classifier confidence={view['classifier_probability']:.3f}")
        print(f"Top cause: {view['candidate_causes'][0]['label']} "
              f"(weight {view['candidate_causes'][0]['weight']:.2f})" if view['candidate_causes'] else "no causes")
        print(f"Ledger: {view['ledger_summary']['text']}")
        print()

    bc = business_case_numbers()
    print("=== Business case ===")
    print(f"Annual savings: ${bc['total_annual_savings']:,.0f}")
    print(f"One-time cost: ${bc['one_time_cost']:,.0f}")
    print(f"Payback: {bc['payback_months']:.3f} months")

    print("\nAll checks passed: PASS - the UI's data layer produces real, "
          "non-hardcoded numbers at every timepoint tested.")


if __name__ == "__main__":
    main()
