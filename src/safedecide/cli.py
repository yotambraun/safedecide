"""Command line: ``safedecide <command>``.

safedecide data        --profile clean
safedecide train       --config configs/base.yaml
safedecide calibrate   --model runs/base/best --alpha 0.05
safedecide benchmark   --model runs/base/best [--http URL] [--phocinae URL]
safedecide export      --model runs/base/best
safedecide card        --model runs/base/best --repo you/safedecide-140m --results results.json
safedecide push        --model runs/base/best --repo you/safedecide-140m
safedecide serve       --model runs/base/best
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _eval_pools(max_eval: int, seed: int, sources):
    from .data import load_jev_distill, load_typed_decisions

    return {
        "jev_test_30k": load_jev_distill("test_set_30k", sources, max_rows=max_eval, seed=seed),
        "jev_ood": load_jev_distill("ood", sources=("openjev_v2",), max_rows=max_eval, seed=seed),
        "typed_decisions_test": load_typed_decisions("test"),
    }


def cmd_data(a):
    from .data import PROFILES, describe, load_profile

    tr = load_profile(a.profile, "train", max_rows=a.max_rows)
    va = load_profile(a.profile, "validation", max_rows=a.max_rows)
    print(
        json.dumps(
            {"profile": a.profile, "streams": PROFILES[a.profile], "train": describe(tr), "validation": describe(va)},
            indent=2,
        )
    )


def cmd_train(a):
    from .data import load_profile
    from .train import TrainConfig, train

    cfg = TrainConfig.from_yaml(a.config, run_name=a.run_name, max_train_rows=a.max_rows)
    tr = load_profile(
        cfg.profile,
        "train",
        max_rows=cfg.max_train_rows,
        seed=cfg.seed,
        typed_decisions_repeat=cfg.typed_decisions_repeat,
    )
    va = load_profile(cfg.profile, "validation", max_rows=cfg.val_max, seed=cfg.seed)
    print(json.dumps(train(cfg, tr, va), indent=2)[:2000])


def cmd_calibrate(a):
    from .data import load_profile
    from .pipeline import calibrate_checkpoint, load_config

    profile = load_config(a.model).get("profile", "clean")
    cal = load_profile(profile, "calibration", max_rows=a.max_rows, seed=a.seed)
    cfg = calibrate_checkpoint(a.model, cal, a.alpha, a.delta, seed=a.seed)
    print(json.dumps({k: cfg[k] for k in ("temperatures", "selective_rule")}, indent=2))


def cmd_benchmark(a):
    from .adapters import LocalAdapter, PhocinaeAdapter, SystemOneHTTPAdapter
    from .data import PROFILES
    from .pipeline import benchmark, save_results, shift_test
    from .report import (
        markdown_table,
        plot_automation,
        plot_automation_vs_alpha,
        plot_reliability,
        plot_risk_coverage,
        plot_validity,
    )

    pools = _eval_pools(a.max_eval, a.seed, PROFILES["clean"])
    adapters = []
    for m in a.model or []:
        adapters.append(LocalAdapter(m))
    for url in a.http or []:
        adapters.append(SystemOneHTTPAdapter(url))
    if a.phocinae:
        adapters.append(PhocinaeAdapter(a.phocinae))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for pool_name, pool in pools.items():
        results = []
        for ad in adapters:
            results.append(benchmark(ad, pool, delta=a.delta))
        save_results(results, str(out / f"{pool_name}.json"))
        plot_risk_coverage(results, str(out / f"{pool_name}_risk_coverage.png"), a.alpha)
        plot_automation(results, str(out / f"{pool_name}_automation.png"), a.alpha)
        plot_automation_vs_alpha(results, str(out / f"{pool_name}_automation_vs_alpha.png"))
        plot_validity(results, str(out / f"{pool_name}_validity.png"), a.alpha)
        for r in results:
            safe = r["model"].replace("/", "_")
            plot_reliability(r, str(out / f"{pool_name}_reliability_{safe}.png"))
        report[pool_name] = markdown_table(results, a.alpha)
    if a.model:
        shift = shift_test(LocalAdapter(a.model[0]), pools["jev_test_30k"], pools["jev_ood"], a.alpha, a.delta)
        (out / "shift.json").write_text(json.dumps(shift, indent=2), encoding="utf-8")
    md = "\n".join(f"### {k}\n\n{v}" for k, v in report.items())
    (out / "leaderboard.md").write_text(md, encoding="utf-8")
    print(md)


def cmd_export(a):
    from .export import check_parity, cpu_latency, export_onnx

    paths = export_onnx(a.model)
    report = {
        "max_abs_logit_diff": {k: check_parity(a.model, p) for k, p in paths.items()},
        "latency": {k: cpu_latency(a.model, p) for k, p in paths.items()},
    }
    Path(a.model, "latency.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({**paths, **report}, indent=2))


def cmd_card(a):
    from .hub import render_model_card

    results = json.loads(Path(a.results).read_text(encoding="utf-8")) if a.results else []
    shift = json.loads(Path(a.shift).read_text(encoding="utf-8")) if a.shift else None
    lat_p = Path(a.model, "latency.json")
    lat = json.loads(lat_p.read_text(encoding="utf-8")) if lat_p.exists() else None
    ablation = Path(a.ablation).read_text(encoding="utf-8") if a.ablation else ""
    card = render_model_card(a.model, a.repo, results, ablation, shift, lat, a.alpha)
    Path(a.model, "README.md").write_text(card, encoding="utf-8")
    print(f"wrote {Path(a.model, 'README.md')}")


def cmd_push(a):
    from .hub import push_model

    card = Path(a.model, "README.md").read_text(encoding="utf-8")
    print(push_model(a.model, a.repo, card, private=a.private))


def cmd_serve(a):
    from .server import serve

    serve(a.model, a.host, a.port)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="safedecide", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("data", help="load and describe a data profile")
    s.add_argument("--profile", default="clean")
    s.add_argument("--max-rows", type=int)
    s.set_defaults(fn=cmd_data)

    s = sub.add_parser("train")
    s.add_argument("--config", required=True)
    s.add_argument("--run-name")
    s.add_argument("--max-rows", type=int)
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("calibrate")
    s.add_argument("--model", required=True)
    s.add_argument("--alpha", type=float, default=0.05)
    s.add_argument("--delta", type=float, default=0.1)
    s.add_argument("--max-rows", type=int, default=8000)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_calibrate)

    s = sub.add_parser("benchmark")
    s.add_argument("--model", action="append", help="local checkpoint (repeatable)")
    s.add_argument("--http", action="append", help="any /v1/systemone server URL (repeatable)")
    s.add_argument("--phocinae", help="phocinae-server URL, e.g. http://127.0.0.1:8155")
    s.add_argument("--alpha", type=float, default=0.05)
    s.add_argument("--delta", type=float, default=0.1)
    s.add_argument("--max-eval", type=int, default=4000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="results")
    s.set_defaults(fn=cmd_benchmark)

    s = sub.add_parser("export")
    s.add_argument("--model", required=True)
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("card")
    s.add_argument("--model", required=True)
    s.add_argument("--repo", required=True)
    s.add_argument("--results")
    s.add_argument("--shift")
    s.add_argument("--ablation", help="markdown file with the ConfTr ablation table")
    s.add_argument("--alpha", type=float, default=0.05)
    s.set_defaults(fn=cmd_card)

    s = sub.add_parser("push")
    s.add_argument("--model", required=True)
    s.add_argument("--repo", required=True)
    s.add_argument("--private", action="store_true")
    s.set_defaults(fn=cmd_push)

    s = sub.add_parser("serve")
    s.add_argument("--model", required=True)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8177)
    s.set_defaults(fn=cmd_serve)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
