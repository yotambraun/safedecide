"""safedecide: small decision models that know when not to decide.

The statistics (``calibrate_selective`` and friends) and the data types import
without torch; ``SafeDecider``, which runs a model, loads torch on first use.
"""

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from .conformal import (
    automation_at_alpha,
    calibrate_selective,
    calibrate_selective_by_case,
    conformal_set_stats,
    one_per_case,
)
from .schema import Decision, decisions_from_systemone_request

if TYPE_CHECKING:
    from .predict import SafeDecider

try:
    __version__ = version("safedecide")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0.0.0"

__all__ = [
    "Decision",
    "SafeDecider",
    "automation_at_alpha",
    "calibrate_selective",
    "calibrate_selective_by_case",
    "conformal_set_stats",
    "decisions_from_systemone_request",
    "one_per_case",
]


def __getattr__(name: str) -> Any:
    if name == "SafeDecider":  # imports torch and transformers, so only when asked for
        from .predict import SafeDecider

        return SafeDecider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), "SafeDecider"])
