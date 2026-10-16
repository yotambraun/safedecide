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

This is the only notebook that needs a paid GPU. It trains the baseline and the Conformal
Training model on the same data and seed, calibrates both, evaluates them, and saves
everything to Google Drive. Export, model card and publishing run in
`02_publish_cpu.ipynb` on a free CPU runtime.

**Budget.** About 25 minutes per training run (two runs) plus about 15 minutes of
setup and evaluation: roughly 70 minutes of GPU time. Set `TRAIN_MINUTES` below to
change it. The runtime disconnects itself at the end so it stops using compute units.

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
SEC_PER_STEP = train(probe, train_set, val_set)["seconds_per_step"]
MAX_STEPS = int(TRAIN_MINUTES * 60 * 0.85 / SEC_PER_STEP)  # 15% headroom for evaluation and checkpoints
print(f"{SEC_PER_STEP:.3f} s/step -> {MAX_STEPS} steps per run")
""",
)
md(gpu, "## Train: baseline, then Conformal Training (same data, same seed)")
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

fig, ax = plt.subplots(figsize=(7, 3.5), dpi=120)
for i, (name, s) in enumerate([("no ConfTr", summary_base), ("ConfTr", summary_ct)]):
    ax.plot([r["step"] for r in s["history"]], [r["nll"] for r in s["history"]], color=SERIES[i], lw=2, label=name)
_style(ax, "Validation NLL", "step", "NLL (lower is better)")
ax.legend(frameon=False)
plt.show()
""",
)
md(
    gpu,
    """
## Calibrate (where the guarantee comes from)

Half of the calibration carve fits one temperature per question type; the other, disjoint
half fits the automate/escalate threshold with exact Clopper–Pearson bounds. The result is
written into `safedecide_config.json` next to the weights.

The model to release is chosen here, on calibration data (most decisions automated), never
on test data.
""",
)
code(
    gpu,
    """
from safedecide.pipeline import calibrate_checkpoint

cal = {path: calibrate_checkpoint(path, calib_set, alpha=ALPHA, delta=DELTA) for path in (BASE_DIR, CT_DIR)}
for path, c in cal.items():
    print(path, json.dumps({"temperatures": c["temperatures"], "rule": c["selective_rule"]}, indent=1))
BEST_DIR = max(cal, key=lambda p: (cal[p]["selective_rule"]["calibration_automation"], -cal[p]["val"]["nll"]))
print("Release candidate:", BEST_DIR)
""",
)
md(gpu, "## Evaluate on every pool")
code(
    gpu,
    """
from IPython.display import Markdown, display

from safedecide.adapters import LocalAdapter
from safedecide.data import typed_decisions_splits
from safedecide.pipeline import benchmark, save_predictions, save_results, shift_test, shipped_rule_check
from safedecide.report import ablation_table, markdown_table
from safedecide.typed_decisions import PriorAdapter, UniformAdapter, score_typed_decisions

RES = f"{OUT}/results"
names = {CT_DIR: "SafeDecide (ConfTr)", BASE_DIR: "SafeDecide (no ConfTr)"}
references = [PriorAdapter(typed_decisions_splits()["train"]), UniformAdapter()]
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
ablation = "\\n\\n".join(
    f"**{pool_name}**\\n\\n" + ablation_table(res[0], res[1], (0.01, 0.02, ALPHA, 0.10, 0.20))
    for pool_name, res in all_results.items()
)
open(f"{RES}/ablation.md", "w").write(ablation)
display(Markdown("### Does Conformal Training help? (paired over identical splits)\\n\\n" + ablation))
""",
)
md(
    gpu,
    """
## Out of the box and under shift

*Shipped threshold*: the threshold saved with the model, applied unchanged to typed-decisions
test. *Shift test*: a threshold calibrated on jev-distill `test_set_30k`, deployed on its `ood`
split. The guarantee only promises anything when deployment data looks like calibration data;
report both results either way.
""",
)
code(
    gpu,
    """
shipped = shipped_rule_check(BEST_DIR, pools["typed_decisions_test"])
shift = shift_test(LocalAdapter(BEST_DIR), pools["jev_test_30k"], pools["jev_ood"], ALPHA, DELTA)
json.dump({"release": BEST_DIR, "shipped_rule_on_typed_decisions_test": shipped, "shift": shift,
           "gpu": GPU, "seconds_per_step": SEC_PER_STEP, "max_steps": MAX_STEPS},
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
    plot_automation_vs_alpha(res, f"{RES}/{pool}_automation_vs_alpha.png")
    plot_risk_coverage(res, f"{RES}/{pool}_risk_coverage.png", ALPHA)
    plot_automation(res, f"{RES}/{pool}_automation.png", ALPHA)
    plot_validity(res, f"{RES}/{pool}_validity.png", ALPHA)
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

HF_USER = "yotambraun"
MODEL_REPO = f"{HF_USER}/safedecide-140m"
RESULTS_REPO = f"{HF_USER}/safedecide-results"
SPACE_REPO = f"{HF_USER}/safedecide"

card = render_model_card(BEST_DIR, MODEL_REPO, results["typed_decisions_test"],
                         ablation_md=open(f"{RES}/ablation.md").read(),
                         shift=json.load(open(f"{RES}/shift.json")), latency=latency, alpha=ALPHA,
                         results_repo=RESULTS_REPO)
release_name = "SafeDecide (ConfTr)" if BEST_DIR.endswith("conftr") else "SafeDecide (no ConfTr)"
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
