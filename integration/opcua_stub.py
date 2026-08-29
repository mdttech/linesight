"""
Not connected to anything. Documents the shape a real deployment's
adapter would have, for the proposal and for a real future build.

A real implementation would use the `asyncua` library against each
station's PLC via OPC UA, reading -- never writing -- tags such as:
    ns=2;s=Station7.PalletPresent
    ns=2;s=Station7.CycleComplete
    ns=2;s=Station7.StationState

Note what's absent even in this sketch: no ns=2;s=...SetSpeed, no
ns=2;s=...Command tag anywhere. That's deliberate, matching base.py --
the interface this would implement has no write method to misuse.
"""
from .base import OTAdapter


class OPCUAStub(OTAdapter):
    def get_event_stream(self, since=None):
        raise NotImplementedError(
            "Stub only, not connected to anything -- see this module's "
            "docstring for the real-world shape this would take."
        )

    def get_state_stream(self, since=None):
        raise NotImplementedError(
            "Stub only, not connected to anything -- see this module's "
            "docstring for the real-world shape this would take."
        )
