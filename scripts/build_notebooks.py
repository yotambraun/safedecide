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
    try:
        token = userdata.get("GH_TOKEN")             # only needed while the GitHub repo is private
        url = f"https://x-access-token:{token}@github.com/{REPO}.git"
    except Exception:
        url = f"https://github.com/{REPO}.git"
    r = subprocess.run(["git", "clone", "-q", url, SRC], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("git clone failed: check the GH_TOKEN secret and that 'Notebook access' is on")
    subprocess.run(["git", "-C", SRC, "remote", "set-url", "origin", f"https://github.com/{REPO}.git"], check=True)
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
in `02_publish_cpu.ipynb` on a free CPU runtime.

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

PROFILE = "clean"          # "clean" -> Apache-2.0 release. "research" adds Jev-labelled rows -> non-commercial.
ALPHA, DELTA = 0.05, 0.10  # error budget on automated decisions; 1 - DELTA = confidence in that promise
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

The model to release is chosen among the stage-2 models here, on calibration data (most
decisions automated), never on test data.
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
cal |= {p: calibrate_checkpoint(p, TD["calibration"], alpha=ALPHA, delta=DELTA, temperature_set=TD["validation"])
        for p in FITTED}
for path, c in cal.items():
    print(path, json.dumps({"temperatures": c["temperatures"], "rule": c["selective_rule"]}, indent=1))
BEST_DIR = max(FITTED, key=lambda p: (cal[p]["selective_rule"]["calibration_automation"], -cal[p]["val"]["nll"]))
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
# 02: CPU (free). Baseline, figures, export, card, publish.
# --------------------------------------------------------------------------- #
cpu: list = []
md(
    cpu,
    """
# SafeDecide 2/2: compare, export, publish (CPU, free)

Runs on a **CPU runtime** (*Runtime → Change runtime type → CPU*), so it costs no compute
units. Reads what notebook 1 saved to Drive.

Secrets needed: `GH_TOKEN` (while the repo is private) and `HF_TOKEN` (a Hugging Face
token with *write* access; only for the publish step). Switch on *Notebook access* for both.
""",
)
code(cpu, SETUP + '%pip install -q "/content/safedecide[all]" "phocinae-server==0.1.7"\n%cd /content/safedecide\n')
code(
    cpu,
    """
import json

from IPython.display import Image, Markdown, display

from safedecide.data import load_typed_decisions

RES = f"{OUT}/results"
info = json.load(open(f"{RES}/run_info.json"))
BEST_DIR = info["release"]
td_test = load_typed_decisions("test")
td_scores = json.load(open(f"{RES}/typed_decisions_official.json"))
results = {p: json.load(open(f"{RES}/{p}.json")) for p in ("typed_decisions_test", "jev_test_30k", "jev_ood")}
ALPHA = 0.05
print("Release candidate:", BEST_DIR)
""",
)
md(
    cpu,
    """
## Baseline: Phocinae-Largha-150M-v1 (same mmBERT-small base)

Called in-process, the way its own typed-decisions harness does it (the HTTP server drops each
question's instructions and criteria). Before its numbers go anywhere, the cell checks that this
harness reproduces what the shipped weights are documented to score (0.9055 accuracy, ECE 0.2519).

Its model card discloses that the shipped weights were trained with 500 of these 2,000 test
decisions (gold targets included); its clean 0.791 comes from a retrained model that is not
published. Its rows are therefore labelled *test-exposed*. About 10–15 minutes on CPU.
""",
)
code(
    cpu,
    """
from huggingface_hub import snapshot_download

from safedecide.adapters import PhocinaeAdapter
from safedecide.pipeline import benchmark, save_predictions
from safedecide.typed_decisions import score_typed_decisions

PUBLISHED_ACCURACY = 0.9055  # shipped weights, per the model's own engine and BENCHMARKS.md
phoc_dir = snapshot_download("Phocinae/Phocinae-Largha-150M-v1", local_dir="/content/largha")
phoc = PhocinaeAdapter.local(phoc_dir, device="cpu")
phoc.name = "Phocinae-Largha-150M-v1 (test-exposed)"
phoc_probs = phoc.predict_proba(td_test)
phoc_scores = score_typed_decisions(phoc_probs, td_test)
acc = phoc_scores["overall"]["accuracy"]
REPRODUCED = abs(acc - PUBLISHED_ACCURACY) <= 0.01
print(f"Phocinae accuracy {acc:.4f} vs published {PUBLISHED_ACCURACY} -> reproduced: {REPRODUCED}")
if REPRODUCED:
    td_scores[phoc.name] = phoc_scores
    save_predictions(RES, "typed_decisions_test", phoc.name, td_test, phoc_probs)
    results["typed_decisions_test"].append(benchmark(phoc, td_test, probs=phoc_probs))
else:
    print("Not reproduced: leave Phocinae out of every table and figure until the gap is understood.")
""",
)
md(cpu, "## Tables and figures")
code(
    cpu,
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
md_parts = ["## typed-decisions (test, 2,000 decisions)\\n\\n" + td_table(td_scores)]
for pool, res in results.items():
    md_parts.append(f"## {pool}: Automation @ {ALPHA:.0%}\\n\\n" + markdown_table(res, ALPHA))
    shown = [r for r in res if not r["model"].startswith("SafeDecide general")]  # stage 1 stays in the tables
    plot_automation_vs_alpha(shown, f"{RES}/{pool}_automation_vs_alpha.png")
    plot_risk_coverage(shown, f"{RES}/{pool}_risk_coverage.png", ALPHA)
    plot_automation(shown, f"{RES}/{pool}_automation.png", ALPHA)
    plot_validity(shown, f"{RES}/{pool}_validity.png", ALPHA)
leaderboard = "\\n\\n".join(md_parts)
open(f"{RES}/leaderboard.md", "w").write(leaderboard)
display(Markdown(leaderboard))
for f in ("automation_vs_alpha", "validity", "risk_coverage"):
    display(Image(f"{RES}/typed_decisions_test_{f}.png"))
""",
)
md(cpu, "## Export ONNX (fp32 + int8), parity, CPU latency")
code(
    cpu,
    """
from safedecide.export import check_parity, cpu_latency, export_onnx

paths = export_onnx(BEST_DIR)
latency = {"max_abs_logit_diff": {k: check_parity(BEST_DIR, p) for k, p in paths.items()},
           "latency": {k: cpu_latency(BEST_DIR, p) for k, p in paths.items()}}
json.dump(latency, open(f"{BEST_DIR}/latency.json", "w"), indent=1)
latency
""",
)
md(cpu, "## Model card\n\nRead it in full. Every number must come from the files above.")
code(
    cpu,
    """
import datetime

from safedecide.hub import render_model_card
from safedecide.typed_decisions import eval_results_yaml

HF_USER = "yotambr"  # Hugging Face username (GitHub is yotambraun)
MODEL_REPO = f"{HF_USER}/safedecide-140m"
RESULTS_REPO = f"{HF_USER}/safedecide-results"
SPACE_REPO = f"{HF_USER}/safedecide"

card = render_model_card(BEST_DIR, MODEL_REPO, results["typed_decisions_test"],
                         ablation_md=open(f"{RES}/ablation.md").read(),
                         shift=json.load(open(f"{RES}/shift.json")), latency=latency, alpha=ALPHA,
                         results_repo=RESULTS_REPO)
release_name = info["release_name"]
eval_yaml = eval_results_yaml(
    td_scores[release_name], source_url=f"https://huggingface.co/datasets/{RESULTS_REPO}", user=HF_USER,
    date=str(datetime.date.today()),
    notes="Fitted on the typed-decisions train split (carved by case into train/validation/calibration); "
          "never trained or calibrated on test. One forward pass per option. ECE: 15 equal-width bins.")
os.makedirs(f"{BEST_DIR}/.eval_results", exist_ok=True)
open(f"{BEST_DIR}/.eval_results/typed-decisions.yaml", "w").write(eval_yaml)
display(Markdown(card))
""",
)
md(
    cpu,
    """
## Publish

Creates the model repo (weights, ONNX, card, eval results), the results dataset (raw
predictions) and the demo Space, all **private**. Make them public on launch day from each
repo's settings. Leave `PUBLISH = False` until you have read the card.
""",
)
code(
    cpu,
    """
PUBLISH = False
PRIVATE = True

if PUBLISH:
    from huggingface_hub import HfApi

    from safedecide.hub import push_model, push_results_dataset

    assets = {f"assets/{f}.png": f"{RES}/typed_decisions_test_{f}.png"
              for f in ("automation_vs_alpha", "validity", "risk_coverage")}
    print(push_model(BEST_DIR, MODEL_REPO, card, private=PRIVATE, extra_files=assets))
    print(push_results_dataset(RES, RESULTS_REPO, MODEL_REPO, private=PRIVATE))

    api = HfApi()
    api.create_repo(SPACE_REPO, repo_type="space", space_sdk="gradio", private=PRIVATE, exist_ok=True)
    api.add_space_variable(SPACE_REPO, "MODEL_REPO", MODEL_REPO)
    api.upload_folder(repo_id=SPACE_REPO, repo_type="space", folder_path="space", commit_message="Upload demo")
    print(f"https://huggingface.co/spaces/{SPACE_REPO}")
""",
)
md(
    cpu,
    """
## Copy the results into the GitHub repo

Download `MyDrive/safedecide/results/*.md` and `*.png` into the repo's `results/` folder,
put the numbers into the README results table, and commit. The raw predictions stay on the Hub.
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
write(cpu, "02_publish_cpu.ipynb")

# Same formatting as the rest of the repo (ruff formats notebook cells too).
subprocess.run([sys.executable, "-m", "ruff", "format", "-q", str(ROOT / "notebooks")], check=True)
