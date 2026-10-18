"""Automation at a guaranteed error for every model scored on typed-decisions test.

    uv run python scripts/automation_board.py --predictions results/predictions/typed_decisions_test

Third-party rows come from the per-decision results Hugging Face published in the
public bucket SaylorTwift/typed-decisions-evals (one request per case, the benchmark's
own scorer). They are downloaded once into --cache and parsed as JSON only. Our rows,
and any other model in --predictions, come from the saved prediction files.

Every row uses the same rules:
* correct = the top option is one of the gold's top options (the bucket's scorer counts
  a tie in the gold as correct, so ours does too here);
* confidence = the probability of the top option;
* 100 random calibration/test halves split by case, delta = 0.1, the calibration
  procedure of docs/METHOD.md (fixed-sequence testing; Bonferroni for comparison).
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

from safedecide.conformal import automation_at_alpha

BUCKET = "SaylorTwift/typed-decisions-evals"
ALPHAS = (0.05, 0.10, 0.20)
DELTA = 0.1
# Models whose cards list typed-decisions as training data. Everything else in the bucket
# is a general decision model scored zero-shot.
FITTED = {"mrm8488/embeddinggemma-2-ft-typed-decision", "nandakishorm/vega-08b-public-intents"}


def download_bucket(cache: Path) -> list[Path]:
    cache.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"https://huggingface.co/api/buckets/{BUCKET}/tree?recursive=true") as r:
        tree = json.load(r)
    paths = []
    for item in tree:
        name = item["path"]
        if not (name.startswith("results/") and name.endswith(".json")) or "smoke" in name:
            continue
        local = cache / Path(name).name
        if not local.exists():
            urllib.request.urlretrieve(f"https://huggingface.co/buckets/{BUCKET}/resolve/{name}", local)
        paths.append(local)
    return sorted(paths)


def bucket_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        d = json.loads(path.read_text(encoding="utf-8"))
        conf, correct, groups = [], [], []
        for case in d["per_case"]:
            for q in case["decisions"]:
                conf.append(q["confidence"])
                correct.append(q["correct"])
                groups.append(case["id"])
        kind = "fitted on train" if d["model"] in FITTED else "general, zero-shot"
        rows.append({"model": d["model"], "kind": kind, "conf": conf, "correct": correct, "groups": groups})
    return rows


def prediction_rows(folder: Path) -> list[dict]:
    rows = []
    for path in sorted(folder.glob("*.jsonl")):
        if path.stem == "uniform-reference":  # every option tied: no top option to act on
            continue
        recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        conf, correct = [], []
        for r in recs:
            p, t = np.asarray(r["probs"]), np.asarray(r["target"])
            conf.append(float(p.max()))
            correct.append(int(p.argmax()) in set(np.flatnonzero(np.isclose(t, t.max()))))
        if path.stem.startswith("safedecide"):
            kind = "ours, fitted on train" if "general" not in path.stem else "ours, general"
        elif "test-exposed" in path.stem:
            kind = "fitted, test-exposed"
        else:
            kind = "reference"
        groups = [r["id"].split(":")[0] for r in recs]  # one case = one state, five questions
        rows.append({"model": recs[0]["model"], "kind": kind, "conf": conf, "correct": correct, "groups": groups})
    return rows


def score(row: dict) -> dict:
    out = {
        "model": row["model"],
        "kind": row["kind"],
        "n": len(row["conf"]),
        "accuracy": float(np.mean(row["correct"])),
    }
    for alpha in ALPHAS:
        for method, kw in (("fixed_sequence", {}), ("bonferroni", {"starts": None})):
            a = automation_at_alpha(row["conf"], row["correct"], alpha, DELTA, groups=row["groups"], **kw)
            out[f"{method}@{alpha}"] = {
                k: a[k]
                for k in ("automation_rate_mean", "automation_rate_std", "selective_error_mean", "violation_rate")
            }
    return out


def markdown(board: list[dict]) -> str:
    lines = [
        "| Model | Training | Accuracy | Automation @ 5% | @ 10% | @ 20% | @ 10%, Bonferroni |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in board:
        cells = [f"{r[f'fixed_sequence@{a}']['automation_rate_mean']:.1%}" for a in ALPHAS]
        cells[1] += f" ± {r['fixed_sequence@0.1']['automation_rate_std']:.1%}"
        lines.append(
            f"| {r['model']} | {r['kind']} | {r['accuracy']:.3f} | "
            + " | ".join(cells)
            + f" | {r['bonferroni@0.1']['automation_rate_mean']:.1%} |"
        )
    return "\n".join(lines)


def plot(board: list[dict], path: Path, alpha: float = 0.10) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from safedecide.report import SERIES, SURFACE, TEXT, _style

    colour = {
        "general, zero-shot": SERIES[0],
        "fitted on train": SERIES[3],
        "ours, fitted on train": SERIES[1],
        "fitted, test-exposed": "#b3b1ab",
        "reference": "#b3b1ab",
    }
    shown = [r for r in board if r["kind"] != "ours, general"]
    rows = [r for r in shown if r[f"fixed_sequence@{alpha}"]["automation_rate_mean"] >= 0.005][::-1]
    vals = [r[f"fixed_sequence@{alpha}"]["automation_rate_mean"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(rows) + 1.6), dpi=160)
    ax.barh([r["model"] for r in rows], vals, color=[colour[r["kind"]] for r in rows], height=0.6)
    for y, v in enumerate(vals):
        ax.text(v + 0.01, y, f"{v:.0%}", va="center", color=TEXT, fontsize=8)
    hidden = len(shown) - len(rows)
    xlabel = "Share of typed-decisions test decisions" + (f"  ({hidden} more models: under 0.5%)" if hidden else "")
    _style(ax, f"Automated with a guaranteed error ≤ {alpha:.0%}", xlabel, "")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlim(0, 1.05)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colour.values()][:4]
    ax.legend(handles, list(colour)[:4], loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--predictions", type=Path, help="folder of saved typed_decisions_test predictions (*.jsonl)")
    ap.add_argument("--cache", type=Path, default=Path(".cache/typed-decisions-evals"))
    ap.add_argument("--out", type=Path, default=Path("results/automation_board"), help="writes <out>.json/.md/.png")
    a = ap.parse_args()

    rows = bucket_rows(download_bucket(a.cache))
    if a.predictions:
        rows += prediction_rows(a.predictions)
    board = sorted(
        (score(r) for r in rows), key=lambda r: (-r["fixed_sequence@0.1"]["automation_rate_mean"], -r["accuracy"])
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(board, indent=1), encoding="utf-8")
    table = markdown(board)
    a.out.with_suffix(".md").write_text(table + "\n", encoding="utf-8")
    plot(board, a.out.with_suffix(".png"))
    print(table)


if __name__ == "__main__":
    main()
