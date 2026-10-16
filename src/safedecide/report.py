"""Leaderboard table + figures for the README, model card and launch posts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

# Series colours, assigned in a fixed order so a model keeps its colour across figures.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, color=TEXT_2)
    ax.set_ylabel(ylabel, color=TEXT_2)
    ax.tick_params(colors=TEXT_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)


def _ci(r: dict, metric: str, fmt: str) -> str:
    lo_hi = r.get("ci95", {}).get(metric)
    return f" [{lo_hi[0]:{fmt}}, {lo_hi[1]:{fmt}}]" if lo_hi else ""


def markdown_table(results: Sequence[dict], alpha: float = 0.05) -> str:
    """Leaderboard sorted by Automation @ alpha. Brackets are 95% cluster-bootstrap intervals."""
    key = f"alpha={alpha}"
    head = (
        f"| Model | n | Accuracy | ECE ↓ | NLL ↓ | Automation @ {alpha:.0%} ↑ | "
        f"Realised error | Violation rate |\n|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in sorted(results, key=lambda r: -r["automation"][key]["automation_rate_mean"]):
        m, a = r["metrics"], r["automation"][key]
        rows.append(
            f"| {r['model']} | {m['n']} | {m['accuracy']:.3f}{_ci(r, 'accuracy', '.3f')} | "
            f"{m['ece']:.3f}{_ci(r, 'ece', '.3f')} | {m['nll']:.3f} | "
            f"**{a['automation_rate_mean']:.1%}** ± {a['automation_rate_std']:.1%} | "
            f"{a['selective_error_mean']:.2%} | {a['violation_rate']:.0%} |"
        )
    return head + "\n".join(rows) + "\n"


def ablation_table(with_conftr: dict, without_conftr: dict, alphas: Sequence[float]) -> str:
    """ConfTr vs. no ConfTr on the same pool. The cal/test splits are identical for both
    models, so the automation difference is paired split by split."""
    rows = [
        "| α | Automation without ConfTr | Automation with ConfTr | Paired Δ (mean ± std over splits) |",
        "|---|---|---|---|",
    ]
    for a in alphas:
        k = f"alpha={a}"
        x = np.array(without_conftr["automation"][k]["per_split"]["automation_rate"])
        y = np.array(with_conftr["automation"][k]["per_split"]["automation_rate"])
        d = y - x
        rows.append(f"| {a:.0%} | {x.mean():.1%} | {y.mean():.1%} | {d.mean():+.1%} ± {d.std():.1%} |")
    return "\n".join(rows) + "\n"


def plot_risk_coverage(results: Sequence[dict], path: str, alpha: float = 0.05) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
    for i, r in enumerate(results[: len(SERIES)]):
        cov, err = r["risk_coverage"]["coverage"], r["risk_coverage"]["error"]
        ax.plot(cov, err, color=SERIES[i], linewidth=2, label=r["model"])
        ax.annotate(
            r["model"],
            (cov[-1], err[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            color=TEXT_2,
            fontsize=8,
            va="center",
        )
    ax.axhline(alpha, color=TEXT_2, linestyle="--", linewidth=1)
    ax.text(0.01, alpha, f" error budget {alpha:.0%}", color=TEXT_2, fontsize=8, va="bottom")
    _style(
        ax, "How much can each model decide alone?", "Share of decisions automated", "Error among automated decisions"
    )
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_automation(results: Sequence[dict], path: str, alpha: float = 0.05) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = f"alpha={alpha}"
    rs = sorted(results, key=lambda r: r["automation"][key]["automation_rate_mean"])
    vals = [r["automation"][key]["automation_rate_mean"] for r in rs]
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(rs) + 1.2), dpi=160)
    ax.barh([r["model"] for r in rs], vals, color=SERIES[0], height=0.55)
    for y, v in enumerate(vals):
        ax.text(v + 0.01, y, f"{v:.0%}", va="center", color=TEXT, fontsize=9)
    _style(ax, f"Decisions automated with a guaranteed error ≤ {alpha:.0%}", "Share of decisions", "")
    ax.set_xlim(0, 1.0)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_reliability(result: dict, path: str, n_bins: int = 15) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .metrics import reliability_curve

    conf = np.asarray(result["_conf"])
    correct = np.asarray(result["_correct"], dtype=float)
    xs, ys, ns = reliability_curve(conf, correct, n_bins)
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=160)
    ax.plot([0, 1], [0, 1], color=TEXT_2, linestyle="--", linewidth=1)
    ax.plot(xs, ys, color=SERIES[0], linewidth=2, marker="o", markersize=4)
    _style(ax, f"Reliability: {result['model']}", "Stated confidence", "Observed accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_automation_vs_alpha(results: Sequence[dict], path: str) -> str:
    """Headline figure: share of decisions automated at each guaranteed error budget."""
    import matplotlib
    import matplotlib.ticker

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
    for i, r in enumerate(results[: len(SERIES)]):
        alphas = sorted(float(k.split("=")[1]) for k in r["automation"])
        mean = np.array([r["automation"][f"alpha={a}"]["automation_rate_mean"] for a in alphas])
        std = np.array([r["automation"][f"alpha={a}"]["automation_rate_std"] for a in alphas])
        ax.plot(alphas, mean, color=SERIES[i], linewidth=2, marker="o", markersize=4, label=r["model"])
        ax.fill_between(alphas, mean - std, mean + std, color=SERIES[i], alpha=0.15, linewidth=0)
    _style(ax, "Decisions automated at a guaranteed error budget", "Error budget α", "Share of decisions automated")
    all_alphas = sorted({float(k.split("=")[1]) for r in results for k in r["automation"]})
    ax.set_xticks(all_alphas)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_validity(results: Sequence[dict], path: str, alpha: float = 0.05) -> str:
    """Realised error on the held-out half of every random split. The guarantee says
    the true error exceeds α with probability ≤ δ, so nearly all mass should sit left of α."""
    import matplotlib
    import matplotlib.ticker

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = f"alpha={alpha}"
    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=160)
    errs_all = [np.array(r["automation"][key]["per_split"]["selective_error"]) for r in results]
    hi = max(alpha * 1.6, max((e.max() for e in errs_all if len(e)), default=alpha))
    bins = np.linspace(0, hi, 30)
    for i, (r, errs) in enumerate(zip(results[: len(SERIES)], errs_all)):
        ax.hist(errs, bins=bins, color=SERIES[i], alpha=0.55, label=r["model"])
    ax.axvline(alpha, color=TEXT, linestyle="--", linewidth=1)
    ax.text(alpha, ax.get_ylim()[1] * 0.95, f"  α = {alpha:.0%}", color=TEXT, fontsize=8, va="top")
    _style(ax, "Realised error on held-out data, per random split", "Error among automated decisions", "Splits")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path
