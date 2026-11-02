"""Check an installed safedecide (the release wheel in a fresh venv) without downloading a model.

    python scripts/smoke_installed.py

Fails if the package was imported from a source checkout instead of the install, if
package data (the model card template, the typing marker) is missing from the wheel,
or if the card, the labelled-JSONL reader or the certification no longer work.
"""

from __future__ import annotations

import json
import re
import tempfile
from importlib import resources
from pathlib import Path

import numpy as np

import safedecide

where = Path(safedecide.__file__).resolve()
assert "site-packages" in where.parts or "dist-packages" in where.parts, f"imported from {where}, not the install"
assert safedecide.__version__ != "0.0.0", "no installed package metadata"

package = resources.files("safedecide")
for name in ("model_card.md", "py.typed"):
    assert package.joinpath(name).is_file(), f"{name} is missing from the wheel"

from safedecide.hub import render_model_card  # noqa: E402
from safedecide.pipeline import certify_rules  # noqa: E402
from safedecide.schema import case_of, labelled_decisions_from_jsonl  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    rule = {"alpha": 0.1, "delta": 0.1, "threshold": 0.8, "calibration_automation": 0.4, "statement": "promise"}
    config = {"profile": "clean", "base_model": "base", "selective_rule": rule, "selective_rules": [rule]}
    Path(tmp, "safedecide_config.json").write_text(json.dumps(config), encoding="utf-8")
    card = render_model_card(tmp, "someone/model", [])
    assert "someone/model" in card and "promise" in card
    assert not re.search(r"\$\{?[A-Za-z_]", card), "unfilled placeholder in the packaged model card"

    lines = [
        {"state": f"ticket {i // 2}", "question": {"type": "noul", "instructions": "urgent?"}, "answer": i % 3 == 0}
        for i in range(40)
    ]
    data = Path(tmp, "mine.jsonl")
    data.write_text("\n".join(json.dumps(r) for r in lines), encoding="utf-8")
    decisions = labelled_decisions_from_jsonl(str(data))
    assert len(decisions) == 40

rng = np.random.default_rng(0)
rules, info = certify_rules(
    rng.uniform(0.5, 1, 40), rng.uniform(0, 1, 40) < 0.9, [case_of(d) for d in decisions], 0.1, 0.1, unit="one-per-case"
)
assert info["n_selective"] == 20 and len(rules) == 3
rules, info = certify_rules(
    rng.uniform(0.5, 1, 40), rng.uniform(0, 1, 40) < 0.9, [case_of(d) for d in decisions], 0.1, 0.1, unit="case"
)
assert info == {"n_selective": 40, "n_cases": 20, "unit": "case"} and len(rules) == 3

print(f"safedecide {safedecide.__version__} from {where.parent}: package data and offline API ok")
