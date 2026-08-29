"""
Tracks predictions against outcomes, in memory -- no database needed for
a demo run. This is the direct answer to "predictive claims must be
validated against real outcomes over time": every prediction the twin
makes gets logged, and once the outcome is known, it's marked confirmed
or false_alarm. A rolling accuracy figure sits next to every new
recommendation, so trust is earned from a visible track record rather
than assumed.
"""
import uuid
from datetime import datetime, timedelta


class PredictionLedger:
    def __init__(self):
        self._predictions = {}  # id -> dict
        self._order = []        # insertion order, oldest first

    def log(self, station, kind, predicted_for, confidence, made_at=None):
        """Logs a new prediction. predicted_for: the timestamp the
        prediction is ABOUT (e.g. 'station 7 will be the bottleneck at
        this time'). Returns the prediction's id."""
        pred_id = str(uuid.uuid4())
        self._predictions[pred_id] = {
            "id": pred_id,
            "station": station,
            "kind": kind,
            "made_at": made_at or datetime.utcnow(),
            "predicted_for": predicted_for,
            "confidence": confidence,
            "status": "pending",
        }
        self._order.append(pred_id)
        return pred_id

    def resolve(self, pred_id, outcome_occurred: bool):
        """Marks a pending prediction confirmed or false_alarm. Raises
        KeyError for an unknown id and ValueError if it's already
        resolved -- resolving twice would silently corrupt the accuracy
        figure, so this fails loudly instead."""
        if pred_id not in self._predictions:
            raise KeyError(f"No prediction with id {pred_id}")
        pred = self._predictions[pred_id]
        if pred["status"] != "pending":
            raise ValueError(f"Prediction {pred_id} already resolved as {pred['status']}")
        pred["status"] = "confirmed" if outcome_occurred else "false_alarm"

    def rolling_accuracy(self, kind=None, window=20):
        """Fraction confirmed among the most recent `window` RESOLVED
        predictions (pending ones don't count either way), most recent
        first. Returns None if there's nothing resolved yet to report."""
        resolved = [
            self._predictions[pid] for pid in reversed(self._order)
            if self._predictions[pid]["status"] != "pending"
            and (kind is None or self._predictions[pid]["kind"] == kind)
        ][:window]
        if not resolved:
            return None
        confirmed = sum(1 for p in resolved if p["status"] == "confirmed")
        return confirmed / len(resolved)

    def summary(self, kind=None, window=20):
        """Human-readable 'N of M confirmed' plus the raw numbers, for
        the UI widget."""
        resolved = [
            self._predictions[pid] for pid in reversed(self._order)
            if self._predictions[pid]["status"] != "pending"
            and (kind is None or self._predictions[pid]["kind"] == kind)
        ][:window]
        confirmed = sum(1 for p in resolved if p["status"] == "confirmed")
        false_alarms = len(resolved) - confirmed
        pending = sum(1 for pid in self._order
                      if self._predictions[pid]["status"] == "pending"
                      and (kind is None or self._predictions[pid]["kind"] == kind))
        return {
            "confirmed": confirmed,
            "false_alarms": false_alarms,
            "pending": pending,
            "total_resolved": len(resolved),
            "text": f"last {len(resolved)} predictions: {confirmed} confirmed, "
                    f"{false_alarms} false alarm{'s' if false_alarms != 1 else ''}"
                    + (f", {pending} pending" if pending else ""),
        }

    def all_predictions(self):
        return [self._predictions[pid] for pid in self._order]
