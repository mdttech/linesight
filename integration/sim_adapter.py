"""
The only OTAdapter actually connected to anything -- reads the plant's
own CSV output. Streams rather than loading everything into a list, the
same way a real integration reading off a live event bus would.
"""
import csv
from datetime import datetime

from .base import OTAdapter

TS_FORMAT = "%Y-%m-%d %H:%M:%S"


def _parse_ts(s):
    return datetime.strptime(s, TS_FORMAT)


class SimAdapter(OTAdapter):
    def __init__(self, out_dir):
        self.event_log_path = f"{out_dir}/event_log.csv"
        self.state_log_path = f"{out_dir}/state_log.csv"

    def get_event_stream(self, since=None):
        with open(self.event_log_path, newline="") as f:
            for row in csv.DictReader(f):
                ts_finish = _parse_ts(row["ts_finish"])
                if since is not None and ts_finish <= since:
                    continue
                yield {
                    "part_id": int(row["part_id"]),
                    "activity": int(row["activity"]),
                    "ts_start": _parse_ts(row["ts_start"]),
                    "ts_finish": ts_finish,
                    "result": row["result"],
                    "scrap": row["scrap"],
                }

    def get_state_stream(self, since=None):
        with open(self.state_log_path, newline="") as f:
            for row in csv.DictReader(f):
                ts_end = _parse_ts(row["ts_end"])
                if since is not None and ts_end <= since:
                    continue
                yield {
                    "station": int(row["station"]),
                    "state": row["state"],
                    "ts_start": _parse_ts(row["ts_start"]),
                    "ts_end": ts_end,
                }
