"""Generate the Colab notebooks from the cell lists below.

    uv run python scripts/build_notebooks.py

Keeping the source here (not hand-edited .ipynb JSON) makes notebook changes reviewable.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SETUP = r"""
import os
import subprocess

from google.colab import drive, userdata

drive.mount("/content/drive")
OUT = "/content/drive/MyDrive/safedecide"           # everything worth keeping goes here
os.makedirs(f"{OUT}/runs", exist_ok=True)
os.makedirs(f"{OUT}/results", exist_ok=True)

REPO = "yotambraun/safedecide"
SRC = "/content/safedecide"
if not os.path.exists(SRC):
    public = f"https://github.com/{REPO}.git"
    branch = ["-b", BRANCH] if globals().get("BRANCH") else []  # set by notebooks that run a branch
    r = subprocess.run(["git", "clone", "-q", *branch, public, SRC], capture_output=True, text=True)
    if r.returncode:                                 # the GitHub repo was private before launch
        token = userdata.get("GH_TOKEN")
        url = f"https://x-access-token:{token}@github.com/{REPO}.git"
        r = subprocess.run(["git", "clone", "-q", *branch, url, SRC], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"git clone failed: {r.stderr.strip()}")
    subprocess.run(["git", "-C", SRC, "remote", "set-url", "origin", public], check=True)
print(subprocess.run(["git", "-C", SRC, "log", "-1", "--format=%h %s"], capture_output=True, text=True).stdout)
"""


def md(cells, s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})


def code(cells, s):
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": s.strip("\n").splitlines(keepends=True),
        }
    )


# --------------------------------------------------------------------------- #
# 01: GPU (paid). Train, calibrate, evaluate. Nothing else.
# --------------------------------------------------------------------------- #
gpu: list = []
md(
    gpu,
    """
# SafeDecide 1/2: train, calibrate, evaluate (GPU)

This is the only notebook that needs a paid GPU. Stage 1 trains a general model on
jev-distill plus typed-decisions, with and without Conformal Training. Stage 2 fits it to
typed-decisions alone, again with and without Conformal Training. Every model is calibrated
and evaluated, and everything is saved to Google Drive. Export, model card and publishing run
in `02_publish.ipynb` (a T4 for about 20 minutes, or CPU in about three hours).

**Budget.** About 25 minutes per stage-1 run (two runs), 5 minutes per stage-2 run (three
runs) and 15 minutes of setup and evaluation: roughly 80 minutes of GPU time. Set
`TRAIN_MINUTES` below to change stage 1. Runs already on Drive are reused, not retrained.
The runtime disconnects itself at the end so it stops using compute units.

**Before you run.**
1. *Runtime → Change runtime type → A100 GPU* (L4 also works; T4 uses a lighter config).
2. Key icon (Secrets) on the left: add `GH_TOKEN` (a fine-grained GitHub token with read
   access to this repo; only needed while it is private) and switch on *Notebook access*.
3. *Runtime → Run all*. If the runtime drops, run all again: finished training runs are
   found on Drive and skipped.
""",
)
code(gpu, SETUP + '%pip install -q "/content/safedecide[all]"\n%cd /content/safedecide\n')
md(gpu, "## Settings")
code(
    gpu,
    """
import json
import shutil
import time
from pathlib import Path

import torch
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()  # one "Loading weights" bar per model load buries the output

PROFILE = "clean"          # "clean" -> Apache-2.0 release. "research" adds Jev-labelled rows -> non-commercial.
ALPHA, DELTA = 0.05, 0.10  # error budget on automated decisions; 1 - DELTA = confidence in that promise
TD_ALPHA = 0.10            # default level of the typed-decisions release (5%, 10% and 20% are all certified)
TRAIN_MINUTES = 25         # per training run (baseline and ConfTr)
MAX_EVAL = 4000            # decisions per jev-distill evaluation pool
DISCONNECT_WHEN_DONE = True

assert torch.cuda.is_available(), "No GPU: Runtime > Change runtime type > A100 (or L4)"
GPU = torch.cuda.get_device_name(0)
CONFIG = "configs/base.yaml" if torch.cuda.get_device_capability(0)[0] >= 8 else "configs/colab_t4.yaml"
print(GPU, "->", CONFIG)
T0 = time.time()
""",
)
md(
    gpu,
    """
## Data

Training: jev-distill `openjev_v2` (CC0 at source) plus typed-decisions `train`, carved by
case into train / validation / calibration. typed-decisions `test` is only ever used for
the final evaluation.
""",
)
code(
    gpu,
    """
from safedecide.data import PROFILES, describe, load_jev_distill, load_profile, load_typed_decisions
from safedecide.train import TrainConfig, train

base_cfg = TrainConfig.from_yaml(CONFIG, profile=PROFILE)
train_set = load_profile(PROFILE, "train", max_rows=base_cfg.max_train_rows, seed=base_cfg.seed,
                         typed_decisions_repeat=base_cfg.typed_decisions_repeat)
val_set = load_profile(PROFILE, "validation", max_rows=base_cfg.val_max, seed=base_cfg.seed)
calib_set = load_profile(PROFILE, "calibration", max_rows=8000)
pools = {
    "typed_decisions_test": load_typed_decisions("test"),  # the community benchmark (headline)
    "jev_test_30k": load_jev_distill("test_set_30k", PROFILES["clean"], max_rows=MAX_EVAL),
    "jev_ood": load_jev_distill("ood", ("openjev_v2",), max_rows=MAX_EVAL),
}
print(json.dumps({"train": describe(train_set), "validation": describe(val_set),
                  "calibration": describe(calib_set), **{k: describe(v)["n"] for k, v in pools.items()}}, indent=1))
""",
)
md(
    gpu,
    "## Size the runs to the time budget\n\n40 steps to measure this GPU's speed; each run then gets as many steps as fit in `TRAIN_MINUTES`, and the learning-rate schedule is sized to that.",
)
code(
    gpu,
    """
probe = TrainConfig.from_yaml(CONFIG, run_name="probe", profile=PROFILE, output_dir="/content/probe")
probe.max_steps, probe.eval_every, probe.val_max = 40, 10**9, 64
probe_summary = train(probe, train_set, val_set)
SEC_PER_STEP = probe_summary["train_step_seconds"] or probe_summary["seconds_per_step"]  # start-up excluded
MAX_STEPS = int(TRAIN_MINUTES * 60 * 0.85 / SEC_PER_STEP)  # 15% headroom for evaluation and checkpoints
print(f"{SEC_PER_STEP:.3f} s/step -> {MAX_STEPS} steps per run")
""",
)
md(gpu, "## Stage 1: general training (baseline, then Conformal Training; same data, same seed)")
code(
    gpu,
    """
def run(name, **overrides):
    done = f"{OUT}/runs/{name}"
    if os.path.exists(f"{done}/train_summary.json"):
        print(f"{name}: already trained, using {done}")
        return done, json.load(open(f"{done}/train_summary.json"))
    cfg = TrainConfig.from_yaml(CONFIG, run_name=name, profile=PROFILE, output_dir="/content/runs", **overrides)
    cfg.max_steps, cfg.max_minutes = MAX_STEPS, TRAIN_MINUTES + 5
    summary = train(cfg, train_set, val_set)
    shutil.copytree(summary["best_dir"], done, dirs_exist_ok=True)  # survives a disconnect
    json.dump(summary, open(f"{done}/train_summary.json", "w"), indent=1)
    return done, summary

BASE_DIR, summary_base = run("safedecide-base")
""",
)
code(gpu, 'CT_DIR, summary_ct = run("safedecide-conftr", w_conftr=0.1, conftr_alpha=ALPHA, conftr_start_frac=0.3)')
code(
    gpu,
    """
import matplotlib.pyplot as plt

from safedecide.report import SERIES, _style


def plot_val_nll(runs, title):
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=120)
    for i, (name, s) in enumerate(runs):
        ax.plot([r["step"] for r in s["history"]], [r["nll"] for r in s["history"]], color=SERIES[i], lw=2, label=name)
    _style(ax, title, "step", "NLL (lower is better)")
    ax.legend(frameon=False)
    plt.show()


plot_val_nll([("no ConfTr", summary_base), ("ConfTr", summary_ct)], "Stage 1: validation NLL")
""",
)
md(
    gpu,
    """
## Stage 2: fit on typed-decisions `train`

typed-decisions gold comes from its own teacher model; jev-distill labels come from a different
one. Stage 2 fine-tunes on the typed-decisions train carve alone, with no input truncated, which
is the setting of that benchmark's "fitted on train" table. Both stage-2 runs start from the
stage-1 baseline with the same data and seed, so Conformal Training is the only difference. A
third run fits typed-decisions alone, straight from mmBERT-small, to measure what stage 1 adds.
About 5 minutes per run on an A100.
""",
)
code(
    gpu,
    """
from safedecide.data import typed_decisions_splits

TD = typed_decisions_splits()  # train / validation / calibration carve of typed-decisions train, by case
TD_CONFIG = "configs/typed_decisions.yaml"
SMALL_GPU = {} if CONFIG == "configs/base.yaml" else {"max_pairs_per_batch": 32, "grad_accum": 2,
                                                      "gradient_checkpointing": True}


def run_td(name, init_from, **overrides):
    done = f"{OUT}/runs/{name}"
    if os.path.exists(f"{done}/train_summary.json"):
        print(f"{name}: already trained, using {done}")
        return done, json.load(open(f"{done}/train_summary.json"))
    cfg = TrainConfig.from_yaml(TD_CONFIG, run_name=name, init_from=init_from, output_dir="/content/runs",
                                **SMALL_GPU, **overrides)
    summary = train(cfg, TD["train"], TD["validation"])
    shutil.copytree(summary["best_dir"], done, dirs_exist_ok=True)
    json.dump(summary, open(f"{done}/train_summary.json", "w"), indent=1)
    return done, summary


TD_DIR, summary_td = run_td("safedecide-td", BASE_DIR)
""",
)
code(gpu, 'TDCT_DIR, summary_tdct = run_td("safedecide-td-conftr", BASE_DIR, w_conftr=0.1, conftr_alpha=ALPHA)')
code(gpu, 'TDONLY_DIR, summary_tdonly = run_td("safedecide-td-only", None, lr=5e-5)')
code(
    gpu,
    """
plot_val_nll([("no ConfTr", summary_td), ("ConfTr", summary_tdct), ("typed-decisions only", summary_tdonly)],
             "Stage 2: typed-decisions validation NLL")
""",
)
md(
    gpu,
    """
## Calibrate (where the guarantee comes from)

One temperature per question type is fitted first; then a disjoint set of decisions fits the
automate/escalate threshold with exact Clopper–Pearson bounds and fixed-sequence testing. The
result is written into `safedecide_config.json` next to the weights. Stage-1 models split the
mixed carve (jev-distill + typed-decisions) in two halves, one per step. Stage-2 models fit
their temperatures on the typed-decisions validation carve, so the whole typed-decisions
calibration carve (600 decisions) sets the threshold.

The model to release is the stage-2 model with the lowest validation NLL. Choosing it on the
calibration carve would reuse the data behind its guarantee, and test data is never used.
Every model gets a threshold at 5%, 10% and 20% error; the typed-decisions release defaults
to 10% (on that benchmark's noisy gold, 600 decisions rarely certify 5%).
""",
)
code(
    gpu,
    """
from safedecide.pipeline import calibrate_checkpoint

GENERAL = {CT_DIR: "SafeDecide general (ConfTr)", BASE_DIR: "SafeDecide general (no ConfTr)"}
FITTED = {TDCT_DIR: "SafeDecide (ConfTr)", TD_DIR: "SafeDecide (no ConfTr)",
          TDONLY_DIR: "SafeDecide, typed-decisions only"}
cal = {p: calibrate_checkpoint(p, calib_set, alpha=ALPHA, delta=DELTA) for p in GENERAL}
# Stage 2: temperatures on the typed-decisions validation carve, so all of the calibration carve sets the threshold
cal |= {p: calibrate_checkpoint(p, TD["calibration"], alpha=TD_ALPHA, delta=DELTA, temperature_set=TD["validation"])
        for p in FITTED}
for path, c in cal.items():
    print(path, json.dumps({"temperatures": c["temperatures"], "rules": c["selective_rules"]}, indent=1))
BEST_DIR = min(FITTED, key=lambda p: cal[p]["val"]["nll"])  # validation only: the calibration carve stays untouched
RELEASE_NAME = FITTED[BEST_DIR]
print("Release candidate:", RELEASE_NAME, BEST_DIR)
""",
)
md(gpu, "## Evaluate on every pool")
code(
    gpu,
    """
from IPython.display import Markdown, display

from safedecide.adapters import LocalAdapter
from safedecide.pipeline import benchmark, save_predictions, save_results, shift_test, shipped_rule_check
from safedecide.report import ablation_table, markdown_table
from safedecide.typed_decisions import PriorAdapter, UniformAdapter, score_typed_decisions

RES = f"{OUT}/results"
names = {**FITTED, **GENERAL}
references = [PriorAdapter(TD["train"]), UniformAdapter()]
all_results, td_scores = {}, {}
for pool_name, pool in pools.items():
    adapters = [LocalAdapter(p, name=n) for p, n in names.items()]
    if pool_name == "typed_decisions_test":
        adapters += references
    results = []
    for ad in adapters:
        probs = ad.predict_proba(pool)
        save_predictions(RES, pool_name, ad.name, pool, probs)
        results.append(benchmark(ad, pool, delta=DELTA, probs=probs))
        if pool_name == "typed_decisions_test":
            td_scores[ad.name] = score_typed_decisions(probs, pool)
    all_results[pool_name] = results
    save_results(results, f"{RES}/{pool_name}.json")
    display(Markdown(f"### {pool_name}\\n\\n" + markdown_table(results, ALPHA)))
json.dump(td_scores, open(f"{RES}/typed_decisions_official.json", "w"), indent=1)
""",
)
code(
    gpu,
    """
from safedecide.typed_decisions import markdown_table as td_table

display(Markdown("### typed-decisions, official metrics\\n\\n" + td_table(td_scores)))
PAIRS = [("stage 2", "SafeDecide (ConfTr)", "SafeDecide (no ConfTr)"),
         ("stage 1", "SafeDecide general (ConfTr)", "SafeDecide general (no ConfTr)")]
parts = []
for pool_name, res in all_results.items():
    by_name = {r["model"]: r for r in res}
    for stage, with_ct, without_ct in PAIRS:
        parts.append(f"**{pool_name}, {stage}**\\n\\n"
                     + ablation_table(by_name[with_ct], by_name[without_ct], (0.01, 0.02, ALPHA, 0.10, 0.20)))
ablation = "\\n\\n".join(parts)
open(f"{RES}/ablation.md", "w").write(ablation)
display(Markdown("### Does Conformal Training help? (paired over identical splits)\\n\\n" + ablation))
""",
)
md(
    gpu,
    """
## Out of the box and under shift

*Shipped threshold*: the threshold saved with the released model (calibrated on the
typed-decisions calibration carve), applied unchanged to typed-decisions test. *Shift test*: a
threshold calibrated on jev-distill `test_set_30k`, deployed on its `ood` split. The guarantee
only promises anything when deployment data looks like calibration data; report both results
either way.
""",
)
code(
    gpu,
    """
shipped = shipped_rule_check(BEST_DIR, pools["typed_decisions_test"])
shift = shift_test(LocalAdapter(BEST_DIR), pools["jev_test_30k"], pools["jev_ood"], ALPHA, DELTA)
json.dump({"release": BEST_DIR, "release_name": RELEASE_NAME, "shipped_rule_on_typed_decisions_test": shipped,
           "shift": shift, "gpu": GPU, "train_step_seconds": SEC_PER_STEP,
           "steps": {Path(p).name: json.load(open(f"{p}/train_summary.json"))["steps"] for p in names}},
          open(f"{RES}/run_info.json", "w"), indent=1)
json.dump(shift, open(f"{RES}/shift.json", "w"), indent=1)
print(json.dumps({"shipped": shipped, "shift": shift}, indent=1))
""",
)
md(gpu, "## Done: free the GPU")
code(
    gpu,
    """
print(f"GPU time used by this notebook: {(time.time() - T0) / 60:.0f} min. Results: {RES}")
if DISCONNECT_WHEN_DONE:
    from google.colab import runtime
    runtime.unassign()
""",
)

# --------------------------------------------------------------------------- #
# 02: T4 (or CPU). Certify, compare, export, card, publish.
# --------------------------------------------------------------------------- #
pub: list = []
md(
    pub,
    """
# SafeDecide 2/2: compare, export and publish

Turns the models and results that notebook 1 saved to Google Drive into a release:

1. **Certify** the release model at 5%, 10% and 20% error.
2. **Baseline**: score Phocinae-Largha-150M-v1 with its own engine.
3. **Results**: leaderboard tables and figures.
4. **Automation board**: every model Hugging Face scored on typed-decisions, under one calibration.
5. **Export** to ONNX (fp32 and int8) and measure CPU latency.
6. **Model card**, rendered from the files above.
7. **Publish** to the Hugging Face Hub, private until launch.

**Runtime:** T4 GPU, about 20 minutes. A CPU runtime works but takes about three hours: on two
cores the long typed-decisions inputs score at about 2 option pairs per second. Latency is
measured on CPU either way.

**Secrets** (🔑 panel, *Notebook access* on): `GH_TOKEN` while the GitHub repo is private, and
`HF_TOKEN` with *write* access for the publish step.
""",
)
md(pub, "## 0. Setup")
code(pub, SETUP + '%pip install -q "/content/safedecide[all]" "phocinae-server==0.1.7"\n%cd /content/safedecide\n')
code(
    pub,
    """
import json
import sys

from IPython.display import Image, Markdown, display
from transformers.utils import logging as hf_logging

from safedecide.data import load_typed_decisions

hf_logging.disable_progress_bar()  # one "Loading weights" bar per model load buries the output

ALPHA, TD_ALPHA, DELTA = 0.05, 0.10, 0.10  # jev-distill level, typed-decisions release default, 1 - confidence
RES = f"{OUT}/results"
info = json.load(open(f"{RES}/run_info.json"))
BEST_DIR = info["release"]
td_test = load_typed_decisions("test")
td_scores = json.load(open(f"{RES}/typed_decisions_official.json"))
results = {p: json.load(open(f"{RES}/{p}.json")) for p in ("typed_decisions_test", "jev_test_30k", "jev_ood")}
print(f"Release model: {info['release_name']} ({BEST_DIR})")
""",
)
md(
    pub,
    """
## 1. Certify the release model

One temperature per question type is fitted on the typed-decisions validation carve. A threshold
is then certified at each error level on the 600-decision calibration carve (exact
Clopper–Pearson bounds, fixed-sequence testing, δ = 10%), and the default one is applied,
unchanged, to the 2,000 test decisions. Skipped when the checkpoint on Drive is already certified.
""",
)
code(
    pub,
    """
from safedecide.data import typed_decisions_splits
from safedecide.hub import rules_table
from safedecide.pipeline import calibrate_checkpoint, shipped_rule_check

cfg = json.load(open(f"{BEST_DIR}/safedecide_config.json"))
if "selective_rules" not in cfg or cfg["selective_rule"]["alpha"] != TD_ALPHA:
    TD = typed_decisions_splits()
    cfg = calibrate_checkpoint(BEST_DIR, TD["calibration"], alpha=TD_ALPHA, delta=DELTA,
                               temperature_set=TD["validation"])
shipped = info["shipped_rule_on_typed_decisions_test"]
if shipped["alpha"] != cfg["selective_rule"]["alpha"]:  # also catches a run interrupted after calibration
    shipped = info["shipped_rule_on_typed_decisions_test"] = shipped_rule_check(BEST_DIR, td_test)
    json.dump(info, open(f"{RES}/run_info.json", "w"), indent=1)

display(Markdown(rules_table(cfg["selective_rules"], cfg["selective_rule"]["alpha"])))
display(Markdown(
    f"**Default threshold on typed-decisions test:** {shipped['automation_rate']:.1%} of "
    f"{shipped['n']:,} decisions automated, {shipped['selective_error']:.1%} of those wrong "
    f"(budget {shipped['alpha']:.0%})."
))
""",
)
md(
    pub,
    """
## 2. Baseline: Phocinae-Largha-150M-v1

Same mmBERT-small base. Called in-process, the way its own typed-decisions harness does it (its
HTTP server drops each question's instructions and criteria). Its numbers are used only if this
harness reproduces the published accuracy of the shipped weights (0.9055) within one point.

Its model card discloses that the shipped weights were trained with 500 of these 2,000 test
decisions, gold targets included; its clean 0.791 comes from a retrained model that is not
published. Its rows are therefore labelled *test-exposed*.
""",
)
code(
    pub,
    """
import logging
import warnings

import torch
from huggingface_hub import snapshot_download

from safedecide.adapters import PhocinaeAdapter
from safedecide.pipeline import benchmark, save_predictions
from safedecide.typed_decisions import score_typed_decisions

# Phocinae compiles its model on GPU; the compiler's notes about that are not errors
torch._logging.set_logs(inductor=logging.ERROR)
warnings.filterwarnings("ignore", module="torch._inductor")

PUBLISHED_ACCURACY = 0.9055  # shipped weights, per the model's own engine and BENCHMARKS.md
phoc_dir = snapshot_download("Phocinae/Phocinae-Largha-150M-v1", local_dir="/content/largha")
phoc = PhocinaeAdapter.local(phoc_dir)  # GPU if there is one
phoc.name = "Phocinae-Largha-150M-v1 (test-exposed)"
phoc_probs = phoc.predict_proba(td_test)
phoc_scores = score_typed_decisions(phoc_probs, td_test)
acc = phoc_scores["overall"]["accuracy"]
REPRODUCED = abs(acc - PUBLISHED_ACCURACY) <= 0.01
pool = [r for r in results["typed_decisions_test"] if r["model"] != phoc.name]  # a rerun replaces, never adds
if REPRODUCED:
    td_scores[phoc.name] = phoc_scores
    save_predictions(RES, "typed_decisions_test", phoc.name, td_test, phoc_probs)
    pool.append(benchmark(phoc, td_test, probs=phoc_probs))
else:
    td_scores.pop(phoc.name, None)
results["typed_decisions_test"] = pool
verdict = "reproduced" if REPRODUCED else "**not reproduced**: left out of every table and figure"
display(Markdown(f"**Phocinae accuracy:** {acc:.4f} (published {PUBLISHED_ACCURACY}), {verdict}."))
""",
)
md(
    pub,
    """
## 3. Results

Automation numbers are means over 100 random calibration/test splits by case; brackets are 95%
bootstrap intervals over cases. typed-decisions is shown at 10% error (the release default),
jev-distill at 5%. The stage-1 ("general") models stay in the tables and are left out of the
figures.
""",
)
code(
    pub,
    """
from safedecide.pipeline import save_results
from safedecide.report import (
    markdown_table,
    plot_automation,
    plot_automation_vs_alpha,
    plot_risk_coverage,
    plot_validity,
)
from safedecide.typed_decisions import markdown_table as td_table

json.dump(td_scores, open(f"{RES}/typed_decisions_official.json", "w"), indent=1)
save_results(results["typed_decisions_test"], f"{RES}/typed_decisions_test.json")
md_parts = ["### typed-decisions test: official metrics (2,000 decisions)\\n\\n" + td_table(td_scores)]
for pool, res in results.items():
    a = TD_ALPHA if pool == "typed_decisions_test" else ALPHA
    md_parts.append(f"### {pool}: automation at a guaranteed error ≤ {a:.0%}\\n\\n" + markdown_table(res, a))
    shown = [r for r in res if not r["model"].startswith("SafeDecide general")]
    plot_automation_vs_alpha(shown, f"{RES}/{pool}_automation_vs_alpha.png")
    plot_risk_coverage(shown, f"{RES}/{pool}_risk_coverage.png", a)
    plot_automation(shown, f"{RES}/{pool}_automation.png", a)
    plot_validity(shown, f"{RES}/{pool}_validity.png", a)
leaderboard = "\\n\\n".join(md_parts)
open(f"{RES}/leaderboard.md", "w").write(leaderboard)
display(Markdown(leaderboard))
""",
)
code(
    pub,
    """
for f in ("automation_vs_alpha", "validity", "risk_coverage"):
    display(Image(f"{RES}/typed_decisions_test_{f}.png", width=900))
""",
)
md(
    pub,
    """
## 4. Automation board

Hugging Face published per-decision results for the models it scored on typed-decisions test
(public bucket `SaylorTwift/typed-decisions-evals`). The same calibration, correctness rule
(a tie in the gold counts as correct, as in that scorer) and case splits are applied to every
row, ours included. Rows are labelled by how each model was trained: zero-shot, fitted on
train, or test-exposed.
""",
)
code(
    pub,
    """
import subprocess

board = f"{RES}/automation_board"
r = subprocess.run([sys.executable, "scripts/automation_board.py",
                    "--predictions", f"{RES}/predictions/typed_decisions_test", "--out", board],
                   capture_output=True, text=True)
if r.returncode:
    raise RuntimeError(r.stderr)
display(Markdown(open(f"{board}.md").read()))
display(Image(f"{board}.png", width=900))
""",
)
md(
    pub,
    """
## 5. Export to ONNX and measure CPU latency

fp32 and int8 exports, each checked against PyTorch on the same inputs, then timed with
onnxruntime on one CPU thread.
""",
)
code(
    pub,
    """
from safedecide.export import check_parity, cpu_latency, export_onnx
from safedecide.hub import latency_table

paths = export_onnx(BEST_DIR)
latency = {"max_abs_logit_diff": {k: check_parity(BEST_DIR, p) for k, p in paths.items()},
           "latency": {k: cpu_latency(BEST_DIR, p) for k, p in paths.items()}}
json.dump(latency, open(f"{BEST_DIR}/latency.json", "w"), indent=1)
display(Markdown(latency_table(latency)))
""",
)
md(
    pub,
    """
## 6. Model card

Rendered from the files above. Read it in full before publishing.
""",
)
code(
    pub,
    """
import datetime

from safedecide.hub import render_model_card
from safedecide.typed_decisions import eval_results_yaml

HF_USER = "yotambr"  # Hugging Face username (GitHub is yotambraun)
EVAL_NOTES = (
    "Fine-tuned on the typed-decisions train split (a general decision model first trained on the openjev_v2 "
    "stream of jev-distill-corpus-v3; train carved by case into train/validation/calibration) - NOT zero-shot; "
    "belongs to the 'fitted or fine-tuned on train' table. Never trained or calibrated on test. Test split, "
    "400 cases / 2,000 decisions, all answered. One encoder pass per option; each question scored on its own "
    "(not all five in one request). Accuracy = agreement with the gold label (0.760 counting a tie in the gold "
    "as correct). ECE = 15-bin max-probability ECE pooled over all decisions (10-bin per question, averaged over "
    "the 20 questions: 0.169)."
)
MODEL_REPO = f"{HF_USER}/safedecide-140m"
RESULTS_REPO = f"{HF_USER}/safedecide-results"
SPACE_REPO = f"{HF_USER}/safedecide"

card = render_model_card(BEST_DIR, MODEL_REPO, results["typed_decisions_test"],
                         ablation_md=open(f"{RES}/ablation.md").read(),
                         shift=json.load(open(f"{RES}/shift.json")), latency=latency, alpha=TD_ALPHA,
                         results_repo=RESULTS_REPO, shipped=shipped)
eval_yaml = eval_results_yaml(
    td_scores[info["release_name"]],
    source_url=f"https://huggingface.co/datasets/{RESULTS_REPO}/blob/main/predictions/typed_decisions_test/"
               "safedecide-no-conftr.jsonl",
    user=HF_USER, date=str(datetime.date.today()), notes=EVAL_NOTES)
os.makedirs(f"{BEST_DIR}/.eval_results", exist_ok=True)
open(f"{BEST_DIR}/.eval_results/typed-decisions.yaml", "w").write(eval_yaml)
display(Markdown(card.split("\\n---\\n", 1)[1]))  # the Hub reads the YAML header; show the page
""",
)
md(
    pub,
    """
## 7. Publish

Creates the model repo (weights, ONNX, card, eval results), the results dataset (raw
predictions) and the in-browser demo Space, all **private**; make them public on launch day from each
repo's settings. After reading the card, set `PUBLISH = True` and run this cell alone.
""",
)
code(
    pub,
    """
PUBLISH = False
PRIVATE = True

if PUBLISH:
    from huggingface_hub import HfApi

    from safedecide.hub import push_model, push_results_dataset

    assets = {f"assets/{f}.png": f"{RES}/typed_decisions_test_{f}.png"
              for f in ("automation_vs_alpha", "validity", "risk_coverage")}
    assets["assets/automation_board.png"] = f"{board}.png"
    print(push_model(BEST_DIR, MODEL_REPO, card, private=PRIVATE, extra_files=assets))
    print(push_results_dataset(RES, RESULTS_REPO, MODEL_REPO, private=PRIVATE))

    api = HfApi()
    # A static Space: the model runs in the visitor's browser (transformers.js), free to host
    api.create_repo(SPACE_REPO, repo_type="space", space_sdk="static", private=PRIVATE, exist_ok=True)
    api.upload_folder(repo_id=SPACE_REPO, repo_type="space", folder_path="space", commit_message="Upload demo")
    print(f"https://huggingface.co/spaces/{SPACE_REPO}")
else:
    print("Nothing uploaded. Read the card above, then set PUBLISH = True and run this cell.")
""",
)
md(
    pub,
    """
## 8. Finish

1. **Runtime → Disconnect and delete runtime.** This notebook leaves the runtime connected so
   you can publish after reading the card; a connected T4 keeps using compute units.
2. Copy `MyDrive/safedecide/results/*.md` and `*.png` into the repo's `results/` folder and
   fill in the README results table. The raw predictions stay on the Hub.
""",
)


# --------------------------------------------------------------------------- #
# 03: GPU (paid). v0.2: choose the recipe by cross-validation, refit, seal, then score test.
# --------------------------------------------------------------------------- #
imp: list = []
md(
    imp,
    """
# SafeDecide v0.2: choose the recipe on train, refit, seal, then score test (GPU)

v0.1's training runs were still improving at their last step (best checkpoint = last step,
for stage 1 and stage 2). This notebook tests, on typed-decisions `train` only, whether
these help, and keeps only what does:

| Recipe | What changes |
|---|---|
| `r0-old-schedule` | v0.1's stage 2 (4 epochs), the baseline |
| `r1-long` | up to 10 epochs, stopped when the validation NLL stops improving |
| `r1-long-s1x2` | the same, on a stage 1 trained twice as long |
| `r2-factors` | the better r1, plus training-only questions about each case's `factors` |
| `+ per-question bias` | one bias per question and option, fitted on held-out predictions |

**Honest by construction.**
- Every choice is made by 5-fold cross-validation on the 1,080 `train` cases outside the
  calibration carve: 5,400 out-of-fold decisions, about ±0.6 accuracy points.
- Inside each fold, a 10% inner carve picks the checkpoint, so the held-out fold is only
  predicted, never used to choose anything.
- Stage 1 never sees typed-decisions in this notebook, so no fold leaks into it.
- The 600 calibration decisions that certified v0.1 stay untouched until the final
  threshold. **`test` is not loaded until `SCORE_TEST = True`.**

**Two passes.**
1. *Pass 1* (`SCORE_TEST = False`): everything up to the final models and a **seal**, the
   SHA-256 of `recipe.json`, which records the choices and the hash of every weight file.
   Post the seal line in discussion #39 before pass 2.
2. *Pass 2* (`SCORE_TEST = True`, Run all again): everything above is reused from Drive. It
   checks the seal, scores `test` once, and writes the results. About 10 minutes.

**What it found** (run on 2026-10-25, sealed and scored on 2026-10-26). Out of fold, the
chosen `r2-factors` reached 0.748 against 0.658 for v0.1's schedule; the bias did not help.
On `test`, the 400M model reached 0.787 (+3.3 points over v0.1, paired 95% interval +1.4
to +5.2) and the 140M v0.2 0.748 (−0.6, no gain). The whole run cost about 25 units.
Details: `docs/METHOD.md`, section 8.

**Budget.** A100 at 5.3 units/hour, as measured on 2026-10-17. Estimated from v0.1's measured
speed: at most about 5 hours and 27 units for the 140M part if every stage-2 run used all 10
epochs. Early stopping usually ends them sooner, so expect less. The cost cell measures this GPU
and prints the upper bound before anything trains. The 400M model is measured separately and
trains only if its upper bound is at most `MAX_UNITS_LARGE`. Every finished run and fold is saved to Drive, so after a disconnect, Run
all continues where it stopped.

**Before you run.**
1. *Runtime → Change runtime type → A100 GPU → Save.*
2. No secrets needed: the code, the data and the base models are all public.
3. Keep this tab open and the computer awake and plugged in.
4. *Runtime → Run all.* At the end the runtime disconnects itself. After that, don't click
   *Connect*, because that starts a new A100.
5. Download `MyDrive/safedecide/v2/report` (no weights in it) and send it.
""",
)
md(imp, "## Setup")
code(
    imp,
    "BRANCH = None  # the default branch; the sealed v0.2 run used commit d4d8ccb\n"
    + SETUP
    + '%pip install -q "/content/safedecide[all]"\n%cd /content/safedecide\n',
)
code(
    imp,
    r"""
import dataclasses
import datetime
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from IPython.display import Markdown, display
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()  # one "Loading weights" bar per model load buries the output

SCORE_TEST = False  # pass 1: False. Pass 2, after posting the seal in discussion #39: True
RUN_LARGE = True  # also fit Ettin-encoder-400m, if its measured cost fits MAX_UNITS_LARGE
MAX_UNITS_LARGE = 12
UNITS_PER_HOUR = 5.3  # A100, from this account's Resources panel (2026-10-17)
PROFILE = "clean"
K = 5
S1_STEPS = {"s1-3355": 3355, "s1-6710": 6710}  # v0.1's stage 1 ran 3,355 steps and was still improving
TD_ALPHA, DELTA = 0.10, 0.10  # the release's default error budget, 1 - confidence in that promise
DISCONNECT_WHEN_DONE = True

assert torch.cuda.is_available(), "No GPU: Runtime > Change runtime type > A100 GPU"
GPU = torch.cuda.get_device_name(0)
if "A100" not in GPU:
    print(f"Note: {GPU}. The unit estimates assume an A100 at {UNITS_PER_HOUR} units/hour.")
V2 = f"{OUT}/v2"  # v0.1's runs and results on Drive are not touched
RUNS, RES, REPORT = f"{V2}/runs", f"{V2}/results", f"{V2}/report"
CV_DIR = f"{REPORT}/cv"  # out-of-fold predictions, small, part of what you download
RECIPE = f"{REPORT}/recipe.json"  # written once, by the seal cell
WORK = "/content/work"  # fold checkpoints: local disk only, deleted after each fold
for p in (RUNS, RES, CV_DIR):
    os.makedirs(p, exist_ok=True)
T0 = time.time()
print(GPU)
""",
)
md(
    imp,
    """
## Data

typed-decisions `train`: five folds by case over the 1,080 cases outside the calibration carve,
plus the carve itself, checked to be the same 600 decisions as v0.1's. Stage 1 uses jev-distill
`openjev_v2` alone (CC0 at source).
""",
)
code(
    imp,
    r"""
from safedecide.data import load_profile, split_by_group, typed_decisions_folds, typed_decisions_splits
from safedecide.factors import factor_decisions, factor_vocabulary

F = typed_decisions_folds(K)
FOLDS, CAL = F["folds"], F["calibration"]
POOL = [d for f in FOLDS for d in f]
assert [d.id for d in CAL] == [d.id for d in typed_decisions_splits()["calibration"]]  # v0.1's 600 decisions
assert {d.state for d in CAL}.isdisjoint({d.state for d in POOL})
VOCAB = factor_vocabulary(POOL)  # which factors become questions, and their options
NEED_STAGE1 = not all(os.path.exists(f"{RUNS}/{n}/train_summary.json") for n in S1_STEPS)
jev_train = load_profile(PROFILE, "train", seed=42, include_typed_decisions=False) if NEED_STAGE1 else []
jev_val = load_profile(PROFILE, "validation", max_rows=3000, seed=42, include_typed_decisions=False) if NEED_STAGE1 else []
print(json.dumps({"folds": [len(f) for f in FOLDS], "calibration": len(CAL), "factor_questions": len(VOCAB),
                  "factor_decisions": len(factor_decisions(POOL, VOCAB)), "stage1_train": len(jev_train),
                  "stage1_validation": len(jev_val)}, indent=1))
""",
)
md(
    imp,
    """
## Cost, measured before anything trains

30 steps of each kind on this GPU, then an upper bound for every run (stage 2 at its full 10
epochs; early stopping usually ends sooner). Runs already on Drive still appear in the table;
they will be reused, not paid again. If the total looks too high, stop the runtime now. Skipped
once everything is sealed.
""",
)
code(
    imp,
    r"""
from safedecide.train import TrainConfig, steps_per_epoch, train


def show_cost(SEC, it):
    v2cfg = TrainConfig.from_yaml("configs/typed_decisions_v2.yaml")
    spe = steps_per_epoch(v2cfg, it)
    spe_aux = steps_per_epoch(dataclasses.replace(v2cfg, aux_fraction=0.5), it, factor_decisions(it, VOCAB))
    spe_full = steps_per_epoch(dataclasses.replace(v2cfg, aux_fraction=0.5), POOL, factor_decisions(POOL, VOCAB))
    plan = {
        "stage 1, 3,355 steps": 3355 * SEC["stage 1"],
        "stage 1, 6,710 steps": 6710 * SEC["stage 1"],
        "CV r0-old-schedule (4 epochs x 5 folds)": K * 4 * spe * SEC["stage 2"],
        "CV r1-long (<= 10 epochs x 5 folds)": K * 10 * spe * SEC["stage 2"],
        "CV r1-long-s1x2 (<= 10 epochs x 5 folds)": K * 10 * spe * SEC["stage 2"],
        "CV r2-factors (<= 10 epochs x 5 folds)": K * 10 * spe_aux * SEC["stage 2"],
        "final 140M refit (<= 10 epochs)": 10 * spe_full * SEC["stage 2"],
    }
    rows = [f"| {k} | {v * OVERHEAD / 60:.0f} | {v * OVERHEAD / 3600 * UNITS_PER_HOUR:.1f} |" for k, v in plan.items()]
    total = sum(plan.values()) * OVERHEAD
    note = "" if SEC["stage 1"] else " Stage 1 is already on Drive (0 here)."
    display(Markdown("| Run | Minutes (upper bound) | Units |\n|---|---|---|\n" + "\n".join(rows)
                     + f"\n| **140M total** | **{total / 60:.0f}** | **{total / 3600 * UNITS_PER_HOUR:.1f}** |"
                     + f"\n\nMeasured: {SEC['stage 1']:.3f} s/step (stage 1), {SEC['stage 2']:.3f} s/step (stage 2)."
                     + note + " The 400M model is measured separately, before it trains."))


def probe(config, data, val, **kw):
    cfg = TrainConfig.from_yaml(config, run_name="probe", profile=PROFILE, output_dir="/content/probe", **kw)
    cfg.max_steps, cfg.epochs, cfg.eval_every, cfg.patience = 30, 1000, 10**9, None  # exactly 30 steps
    s = train(cfg, data, val[:16])
    shutil.rmtree("/content/probe", ignore_errors=True)
    return s["train_step_seconds"] or s["seconds_per_step"]


OVERHEAD = 1.3  # evaluations, model loading, predictions (v0.1 stage 2: 0.305 s per step all-in, 0.22 training)
if os.path.exists(RECIPE):
    print("Everything is trained and sealed: nothing to measure.")
else:
    iv, it = split_by_group([d for f in FOLDS[1:] for d in f], 0.1, seed=0)  # one fold's inner split
    SEC = {"stage 1": probe("configs/base.yaml", jev_train, jev_val) if NEED_STAGE1 else 0.0,
           "stage 2": probe("configs/typed_decisions_v2.yaml", it, iv)}
    show_cost(SEC, it)
""",
)
md(
    imp,
    """
## Stage 1, twice: v0.1's length and twice as long

General decision training on jev-distill alone. The learning-rate schedule is sized to each
run's step count, as in v0.1.
""",
)
code(
    imp,
    r"""
def stage1(name, steps):
    done = f"{RUNS}/{name}"
    if os.path.exists(f"{done}/train_summary.json"):
        print(f"{name}: trained earlier, reused")
        return done
    cfg = TrainConfig.from_yaml("configs/base.yaml", run_name=name, profile=PROFILE, output_dir=WORK, epochs=10)
    cfg.max_steps = steps
    summary = train(cfg, jev_train, jev_val)
    shutil.copytree(summary["best_dir"], done, dirs_exist_ok=True)
    json.dump(summary, open(f"{done}/train_summary.json", "w"), indent=1)  # written last: marks the run done
    return done


S1 = {name: stage1(name, steps) for name, steps in S1_STEPS.items()}
for name, path in S1.items():
    h = json.load(open(f"{path}/train_summary.json"))["history"]
    print(name, "validation NLL by step:", [(r["step"], round(r["nll"], 4)) for r in h[-4:]])
""",
)
md(
    imp,
    """
## Cross-validation, round 1: the schedule and the stage-1 length

Out-of-fold scores, each decision predicted by the one model that never saw its case.
The gain column is paired, on the same decisions, with a 95% bootstrap interval over cases.
""",
)
code(
    imp,
    r"""
from safedecide.cv import comparison_table, cross_validate, oof_probs, score


def v2(init, **kw):
    return TrainConfig.from_yaml("configs/typed_decisions_v2.yaml", init_from=init, output_dir=WORK, **kw)


RECIPES = {  # name -> (config, factor vocabulary or None)
    "r0-old-schedule": (TrainConfig.from_yaml("configs/typed_decisions.yaml", init_from=S1["s1-3355"],
                                              output_dir=WORK, eval_every=100), None),
    "r1-long": (v2(S1["s1-3355"]), None),
    "r1-long-s1x2": (v2(S1["s1-6710"]), None),
}
CV = {name: cross_validate(name, cfg, FOLDS, CV_DIR, WORK, vocab) for name, (cfg, vocab) in RECIPES.items()}
PROBS = {name: oof_probs(cv, POOL) for name, cv in CV.items()}


def acc(name):
    return score(PROBS[name], POOL)["overall"]["accuracy"]


display(Markdown(comparison_table(PROBS, POOL, "r0-old-schedule")))
""",
)
md(
    imp,
    """
## Cross-validation, round 2: factor questions

The better of the two r1 recipes, plus about one training-only factor question for every two
real decisions in each epoch. The held-out fold is scored on its real questions only.
""",
)
code(
    imp,
    r"""
BASE_R1 = max(["r1-long", "r1-long-s1x2"], key=acc)
RECIPES["r2-factors"] = (dataclasses.replace(RECIPES[BASE_R1][0], aux_fraction=0.5), VOCAB)
CV["r2-factors"] = cross_validate("r2-factors", RECIPES["r2-factors"][0], FOLDS, CV_DIR, WORK, VOCAB)
PROBS["r2-factors"] = oof_probs(CV["r2-factors"], POOL)
print(f"r2-factors starts from {BASE_R1}'s stage 1")
display(Markdown(comparison_table(PROBS, POOL, "r0-old-schedule")))
""",
)
md(
    imp,
    """
## Per-question bias, and the choice

The bias is fitted on four folds' out-of-fold predictions and applied to the fifth, so its
gain is measured on cases it never saw. The recipe with the highest out-of-fold accuracy is
chosen. The bias is kept only if it raises that accuracy without raising KL.
""",
)
code(
    imp,
    r"""
from safedecide.cv import crossfit_bias

BEST = max(CV, key=acc)
BIASED = f"{BEST} + per-question bias"
PROBS[BIASED] = crossfit_bias(CV[BEST], POOL)
with_b, without_b = score(PROBS[BIASED], POOL)["overall"], score(PROBS[BEST], POOL)["overall"]
USE_BIAS = with_b["accuracy"] > without_b["accuracy"] and with_b["kl_from_gold"] <= without_b["kl_from_gold"]
SELECTION = comparison_table(PROBS, POOL, "r0-old-schedule") + (
    f"\n**Chosen:** `{BEST}`" + (" + per-question bias" if USE_BIAS else ", without the per-question bias")
    + ". Rule: highest out-of-fold accuracy; the bias only if it raises accuracy without raising KL.\n"
)
open(f"{REPORT}/selection.md", "w").write(SELECTION)
display(Markdown(SELECTION))
""",
)
md(
    imp,
    """
## Final 140M model

The chosen recipe, refit once on all 1,080 cases, stopped at the folds' median best epoch.
Temperatures are the mean of the folds' (each fitted on its inner carve), and the bias, if kept,
is fitted on all out-of-fold predictions. Then the threshold is certified at 5%, 10% and 20% on
the untouched calibration carve.
""",
)
code(
    imp,
    r"""
from safedecide.cv import bias_from_cv, mean_temperatures, refit_step
from safedecide.hub import rules_table
from safedecide.pipeline import calibrate_checkpoint, load_config
from safedecide.predict import CONFIG_NAME


def certified(path):
    return os.path.exists(f"{path}/{CONFIG_NAME}") and "selective_rules" in load_config(path)


def save_config(path, c):
    json.dump(c, open(f"{path}/{CONFIG_NAME}", "w"), indent=2)


def refit_140m(name):
    done = f"{RUNS}/{name}"
    if certified(done):
        print(f"{name}: fitted and certified earlier, reused")
        return done
    cfg, vocab = RECIPES[BEST]
    aux = factor_decisions(POOL, vocab) if vocab else []
    stop = refit_step(CV[BEST], cfg, POOL, aux)
    summary = train(dataclasses.replace(cfg, run_name=name, stop_step=stop, patience=None), POOL, [], aux_set=aux)
    shutil.copytree(summary["best_dir"], done, dirs_exist_ok=True)
    json.dump(summary, open(f"{done}/train_summary.json", "w"), indent=1)
    c = load_config(done)
    c["temperatures"] = mean_temperatures(CV[BEST])
    if USE_BIAS:
        c["question_bias"] = bias_from_cv(CV[BEST], POOL)
    c["recipe"] = {"chosen_by": "5-fold cross-validation on typed-decisions train", "recipe": BEST,
                   "stop_step": stop, "factor_questions": bool(vocab), "question_bias": USE_BIAS,
                   "trained_on": f"typed-decisions train outside the calibration carve ({len({d.state for d in POOL}):,} cases)",
                   "stage1": Path(cfg.init_from).name}
    save_config(done, c)
    calibrate_checkpoint(done, CAL, alpha=TD_ALPHA, delta=DELTA, keep_temperatures=True)
    return done


FINAL = {"SafeDecide-140M v0.2": refit_140m("final-140m")}
cfg140 = load_config(FINAL["SafeDecide-140M v0.2"])
display(Markdown(rules_table(cfg140["selective_rules"], cfg140["selective_rule"]["alpha"])))
""",
)
md(
    imp,
    """
## Optional: Ettin-encoder-400m

The same stage-2 recipe straight from Ettin-encoder-400m (MIT), with no stage 1 and no
cross-validation (5 folds would cost several times more). It trains once on 90% of the
1,080 cases, with early stopping and temperatures on the other 10%, and is certified on the
same calibration carve. It is part of the seal, so it is reported whatever it scores. Its
cost is measured first; it trains only if `RUN_LARGE` and the upper bound is at most
`MAX_UNITS_LARGE` units.
""",
)
code(
    imp,
    r"""
LARGE_DIR, LARGE_SKIP = f"{RUNS}/final-400m", f"{RUNS}/final-400m-skipped.json"
if certified(LARGE_DIR):
    FINAL["SafeDecide-400M v0.2"] = LARGE_DIR
    print("final-400m: fitted and certified earlier, reused")
elif os.path.exists(RECIPE):
    print("Sealed without a 400M model:", open(LARGE_SKIP).read() if os.path.exists(LARGE_SKIP) else "RUN_LARGE was off")
elif RUN_LARGE:
    cfg_l = TrainConfig.from_yaml("configs/typed_decisions_large.yaml", run_name="final-400m", output_dir=WORK,
                                  aux_fraction=RECIPES[BEST][0].aux_fraction)
    vocab_l = RECIPES[BEST][1]
    val_l, train_l = split_by_group(POOL, 0.1, seed=0)
    aux_l = factor_decisions(train_l, vocab_l) if vocab_l else []
    sec_l = probe("configs/typed_decisions_large.yaml", train_l, val_l)
    upper = 10 * steps_per_epoch(cfg_l, train_l, aux_l) * sec_l * OVERHEAD
    units = upper / 3600 * UNITS_PER_HOUR
    print(f"400M: {sec_l:.2f} s/step measured; at most {upper / 60:.0f} min, {units:.1f} units")
    if units > MAX_UNITS_LARGE:
        print(f"Skipped: above MAX_UNITS_LARGE = {MAX_UNITS_LARGE}. Raise it and Run all to fit it.")
        json.dump({"seconds_per_step": sec_l, "upper_bound_units": units, "max_units": MAX_UNITS_LARGE},
                  open(LARGE_SKIP, "w"))
    else:
        summary = train(cfg_l, train_l, val_l, aux_set=aux_l)
        shutil.copytree(summary["best_dir"], LARGE_DIR, dirs_exist_ok=True)
        json.dump(summary, open(f"{LARGE_DIR}/train_summary.json", "w"), indent=1)
        c = load_config(LARGE_DIR)
        c["recipe"] = {"chosen_by": "the 140M model's cross-validated recipe, fixed in advance",
                       "recipe": BEST, "factor_questions": bool(vocab_l), "question_bias": False,
                       "trained_on": f"90% of typed-decisions train outside the calibration carve ({len({d.state for d in train_l}):,} cases)",
                       "stage1": None}
        save_config(LARGE_DIR, c)
        calibrate_checkpoint(LARGE_DIR, CAL, alpha=TD_ALPHA, delta=DELTA, temperature_set=val_l)
        FINAL["SafeDecide-400M v0.2"] = LARGE_DIR
else:
    print("RUN_LARGE is False: no 400M model")
""",
)
md(
    imp,
    """
## Seal

`recipe.json` records the code commit, every out-of-fold score, the choice, and the SHA-256 of
each final model's weights and configuration (thresholds included). Its own SHA-256 is the
seal. Pass 2 refuses to score `test` if any of it has changed.
""",
)
code(
    imp,
    r"""
import subprocess


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


MODELS = {name: {"weights_sha256": sha256(f"{p}/model.safetensors"), "config_sha256": sha256(f"{p}/{CONFIG_NAME}"),
                 "recipe": load_config(p)["recipe"], "drive_dir": p} for name, p in FINAL.items()}
if not os.path.exists(RECIPE):
    doc = {
        "what": "SafeDecide v0.2, typed-decisions: choices made on train only, sealed before test was scored",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "selection": {"out_of_fold": {n: score(p, POOL)["overall"] for n, p in PROBS.items()},
                      "chosen": BEST, "question_bias": USE_BIAS},
        "models": MODELS,
    }
    open(RECIPE, "w").write(json.dumps(doc, indent=1, sort_keys=True))
else:
    sealed = json.load(open(RECIPE))["models"]
    assert sealed == MODELS, "a final model changed after it was sealed; do not score test with it"
SEAL = sha256(RECIPE)
display(Markdown(f"**Seal:** `{SEAL}`\n\nPost this line in discussion #39 before pass 2:\n\n"
                 f"> SafeDecide v0.2 is fixed before scoring test: SHA-256 `{SEAL}` of `recipe.json`, "
                 f"which I'll publish with the results."))
""",
)
md(
    imp,
    """
## Score `test` (pass 2 only)

Once, for each sealed model, with v0.1 next to it for a paired comparison on the same 2,000
decisions (v0.1's predictions come from notebook 1's results on Drive).
""",
)
code(
    imp,
    r"""
if not SCORE_TEST:
    print("test not loaded. Post the seal in #39, then set SCORE_TEST = True and Run all again.")
else:
    from safedecide.adapters import LocalAdapter
    from safedecide.cv import paired_accuracy_delta
    from safedecide.data import load_typed_decisions
    from safedecide.pipeline import benchmark, save_predictions, save_results, shipped_rule_check
    from safedecide.report import markdown_table
    from safedecide.typed_decisions import markdown_table as td_table
    from safedecide.typed_decisions import score_typed_decisions

    assert sha256(RECIPE) == SEAL
    td_test = load_typed_decisions("test")
    probs = {}
    v01 = f"{OUT}/results/predictions/typed_decisions_test/safedecide-no-conftr.jsonl"  # notebook 1's file
    if os.path.exists(v01):
        old = {r["id"]: np.array(r["probs"]) for r in map(json.loads, open(v01))}
        probs["SafeDecide-140M v0.1"] = [old[d.id] for d in td_test]
    shipped, results = {}, []
    for name, path in FINAL.items():
        adapter = LocalAdapter(path, name=name)
        probs[name] = adapter.predict_proba(td_test)
        save_predictions(RES, "typed_decisions_test", name, td_test, probs[name])
        shipped[name] = shipped_rule_check(path, td_test)
    for name, p in probs.items():
        results.append(benchmark(None, td_test, delta=DELTA, probs=p, name=name))
    scores = {name: score_typed_decisions(p, td_test) for name, p in probs.items()}
    base = probs.get("SafeDecide-140M v0.1")
    gains = {name: paired_accuracy_delta(probs[name], base, td_test) for name in FINAL} if base else {}
    json.dump(scores, open(f"{RES}/typed_decisions_official.json", "w"), indent=1)
    save_results(results, f"{RES}/typed_decisions_test.json")
    json.dump({"seal": SEAL, "shipped_rule": shipped, "paired_vs_v0.1": gains}, open(f"{RES}/run_info.json", "w"),
              indent=1)
    report = (f"### typed-decisions test ({len(td_test):,} decisions)\n\n" + td_table(scores)
              + "\n\n### Accuracy gain over v0.1, paired (95% CI over cases)\n\n"
              + "\n".join(f"- {n}: {g['delta']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}]" for n, g in gains.items())
              + "\n\n### Shipped 10% threshold, applied unchanged\n\n"
              + "\n".join(f"- {n}: {s['automation_rate']:.1%} automated, {s['selective_error']:.1%} of those wrong"
                          for n, s in shipped.items())
              + f"\n\n### Automation at a certified error <= {TD_ALPHA:.0%}\n\n" + markdown_table(results, TD_ALPHA))
    open(f"{REPORT}/test_results.md", "w").write(report)
    shutil.copy(f"{RES}/typed_decisions_official.json", REPORT)
    display(Markdown(report))
""",
)
md(imp, "## Done: free the GPU")
code(
    imp,
    r"""
print(f"GPU time used by this notebook: {(time.time() - T0) / 60:.0f} min. Download {REPORT} and send it.")
if DISCONNECT_WHEN_DONE:
    from google.colab import runtime

    runtime.unassign()
""",
)


def write(cells, name, gpu_type=None):
    meta = {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    }
    if gpu_type:
        meta["accelerator"] = "GPU"
        meta["colab"]["gpuType"] = gpu_type
    nb = {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 0}
    path = ROOT / "notebooks" / name
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{path.relative_to(ROOT)}: {len(cells)} cells")


write(gpu, "01_train_gpu.ipynb", gpu_type="A100")
write(pub, "02_publish.ipynb", gpu_type="T4")
write(imp, "03_improve_gpu.ipynb", gpu_type="A100")

# Same formatting as the rest of the repo (ruff formats notebook cells too).
subprocess.run([sys.executable, "-m", "ruff", "format", "-q", str(ROOT / "notebooks")], check=True)
