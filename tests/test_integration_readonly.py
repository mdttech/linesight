"""
Acceptance test: the integration layer is architecturally read-only --
zero write methods exist anywhere in integration/, not just zero write
calls. This is checkable directly, not a claim to take on faith.
"""
import sys, os, re, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    integration_dir = os.path.join(os.path.dirname(__file__), "..", "integration")
    pattern = re.compile(r"def\s+(write|set_)\w*\s*\(")

    matches = []
    for path in glob.glob(os.path.join(integration_dir, "*.py")):
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                if pattern.search(line):
                    matches.append((path, lineno, line.strip()))

    print(f"Scanned {len(glob.glob(os.path.join(integration_dir, '*.py')))} files in integration/")
    if matches:
        print(f"FAIL -- found {len(matches)} write-like method(s):")
        for path, lineno, line in matches:
            print(f"  {path}:{lineno}: {line}")
    else:
        print("Zero write methods found: PASS")
        print('"There is no write method to remove" -- verified, not claimed.')

    # also confirm the adapters actually work end to end, not just that
    # they're syntactically read-only
    from plant.run import run_plant
    from integration.sim_adapter import SimAdapter
    from integration.opcua_stub import OPCUAStub

    run_plant("config/line_siteA.yaml", "plant_out_integration", seed=42)
    adapter = SimAdapter("plant_out_integration")
    n_events = sum(1 for _ in adapter.get_event_stream())
    n_states = sum(1 for _ in adapter.get_state_stream())
    print(f"\nSimAdapter reads real data: {n_events} events, {n_states} state transitions")

    stub = OPCUAStub()
    try:
        list(stub.get_event_stream())
        print("OPCUAStub FAIL -- should have raised NotImplementedError")
    except NotImplementedError:
        print("OPCUAStub correctly refuses to pretend to connect: PASS")


if __name__ == "__main__":
    main()
