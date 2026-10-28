"""Model-specific routes; public callers use :func:`get_adapter`."""

from .base import Adapter, RunRequest
from .mpp import MPPAdapter
from .poseidon import PoseidonAdapter
from .vicon import VICONAdapter


def get_adapter(model: str) -> Adapter:
    adapters: dict[str, Adapter] = {
        "poseidon": PoseidonAdapter(),
        "vicon": VICONAdapter(),
        "mpp": MPPAdapter(),
    }
    try:
        return adapters[model]
    except KeyError as exc:
        raise ValueError(f"Unknown model {model!r}; choose poseidon, vicon, or mpp.") from exc
