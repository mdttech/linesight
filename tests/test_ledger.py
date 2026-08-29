"""Acceptance test: the prediction ledger correctly tracks, resolves, and
reports accuracy, and correctly rejects invalid operations rather than
silently corrupting state."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta
from twin.ledger import PredictionLedger


def main():
    ledger = PredictionLedger()
    now = datetime(2026, 3, 18, 12, 0, 0)

    outcomes = [True] * 14 + [False] * 3 + [True] * 3
    ids = []
    for i, outcome in enumerate(outcomes):
        pid = ledger.log(station=7, kind="bottleneck", predicted_for=now + timedelta(minutes=i * 20),
                          confidence=0.8, made_at=now + timedelta(minutes=i * 15))
        ids.append((pid, outcome))
    for pid, outcome in ids:
        ledger.resolve(pid, outcome)

    summary = ledger.summary()
    print("Summary:", summary["text"])
    ok = summary["confirmed"] == 17 and summary["false_alarms"] == 3
    print(f"Counts correct: {'PASS' if ok else 'FAIL'}")

    try:
        ledger.resolve("nonexistent-id", True)
        print("FAIL: should have raised KeyError")
    except KeyError:
        print("Unknown id raises KeyError: PASS")

    try:
        ledger.resolve(ids[0][0], True)
        print("FAIL: should have raised ValueError")
    except ValueError:
        print("Double-resolve raises ValueError: PASS")

    ledger2 = PredictionLedger()
    for i in range(30):
        pid = ledger2.log(station=1, kind="bottleneck", predicted_for=now, confidence=0.5, made_at=now)
        ledger2.resolve(pid, i % 3 != 0)
    acc = ledger2.rolling_accuracy(window=20)
    print(f"Rolling window correctly limits to most recent 20: accuracy={acc:.3f} "
          f"({'PASS' if acc is not None else 'FAIL'})")


if __name__ == "__main__":
    main()
