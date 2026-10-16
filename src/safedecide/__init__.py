"""safedecide: small decision models that know when not to decide."""

from importlib.metadata import PackageNotFoundError, version

from .conformal import automation_at_alpha, calibrate_selective, conformal_set_stats
from .predict import SafeDecider
from .schema import Decision, decisions_from_systemone_request

try:
    __version__ = version("safedecide")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0.0.0"

__all__ = [
    "Decision",
    "SafeDecider",
    "automation_at_alpha",
    "calibrate_selective",
    "conformal_set_stats",
    "decisions_from_systemone_request",
]
