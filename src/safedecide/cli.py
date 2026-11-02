"""Command line: ``safedecide <command>``.

safedecide data        --profile clean
safedecide train       --config configs/base.yaml
safedecide calibrate   --model yotambr/safedecide-140m --data my_decisions.jsonl --output-dir mine
safedecide calibrate   --http http://localhost:8000 --data my_decisions.jsonl --output rule.json
safedecide decide      --model yotambr/safedecide-140m request.json   (or JSONL; '-' or nothing reads stdin)
safedecide benchmark   --model runs/base/best [--http URL] [--phocinae URL]
safedecide export      --model runs/base/best
safedecide card        --model runs/base/best --repo you/safedecide-140m --results results.json
safedecide push        --model runs/base/best --repo you/safedecide-140m
safedecide serve       --model runs/base/best
safedecide mcp         --model yotambr/safedecide-140m   (MCP server on stdio, for agents)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_RULE_FILE = "safedecide_rule.json"
REVISION_HELP = "Hub branch, tag or commit hash to load (e.g. v0.2); pin a commit for reproducible thresholds"


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


def _calibrate_http(a):
    from . import __version__
    from .adapters import SystemOneHTTPAdapter
    from .pipeline import calibrate_adapter
    from .schema import labelled_decisions_from_jsonl

    if not a.data:
        raise SystemExit("safedecide calibrate: --http needs --data, your labelled decisions")
    for flag, used in (
        ("--output-dir", a.output_dir),
        ("--in-place", a.in_place),
        ("--refit-temperatures", a.refit_temperatures),
        ("--revision", a.revision),
    ):
        if used:
            raise SystemExit(f"safedecide calibrate: {flag} applies to --model, not to --http")
    headers = {}
    for h in a.header or []:
        name, sep, value = h.partition(":")
        if not sep or not name.strip():
            raise SystemExit(f"safedecide calibrate: --header {h!r} is not 'Name: value'")
        headers[name.strip()] = value.strip()
    adapter = SystemOneHTTPAdapter(
        a.http,
        model=a.http_model,
        headers=headers,
        timeout=a.timeout,
        max_concurrency=a.concurrency,
        retries=a.retries,
    )
    result = calibrate_adapter(
        adapter, labelled_decisions_from_jsonl(a.data), a.alpha, a.delta, one_per_case=a.one_per_case, seed=a.seed
    )
    rule_file = {"safedecide_version": __version__, "endpoint": adapter.url, **result}  # headers are not saved
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(rule_file, indent=2), encoding="utf-8")
    print(json.dumps({k: rule_file[k] for k in ("selective_rules", "calibration")}, indent=2))
    print(f"Rule file: {a.output}")


def cmd_calibrate(a):
    from .pipeline import calibrate_checkpoint
    from .schema import labelled_decisions_from_jsonl

    if a.http:
        return _calibrate_http(a)
    if a.output != DEFAULT_RULE_FILE or a.header or a.http_model:
        raise SystemExit("safedecide calibrate: --output, --header and --http-model apply to --http")
    if Path(a.model).is_dir() and not a.output_dir and not a.in_place:
        raise SystemExit(
            f"safedecide calibrate: {a.model} is a local checkpoint. Pass --output-dir DIR to write a "
            "re-calibrated copy, or --in-place to replace the thresholds certified in it."
        )
    if a.in_place and a.output_dir:
        raise SystemExit("safedecide calibrate: choose --output-dir or --in-place, not both")
    if a.data:
        cal = labelled_decisions_from_jsonl(a.data)
    else:  # the calibration split of the model's own training data
        from .data import load_profile
        from .pipeline import load_config

        profile = load_config(a.model).get("profile", "clean")
        cal = load_profile(profile, "calibration", max_rows=a.max_rows, seed=a.seed)
    keep = bool(a.data) and not a.refit_temperatures  # your file goes to the threshold in full
    cfg = calibrate_checkpoint(
        a.model,
        cal,
        a.alpha,
        a.delta,
        seed=a.seed,
        output_dir=a.output_dir,
        keep_temperatures=keep,
        revision=a.revision,
        one_per_case=a.one_per_case,
    )
    print(json.dumps({k: cfg[k] for k in ("temperatures", "selective_rules")}, indent=2))
    print(f"Calibrated copy: {cfg['path']}")


def _read_requests(source: str) -> tuple[list[tuple[int, dict]], bool]:
    """``/v1/systemone`` requests from a file or stdin: one JSON object, or one per line.

    Returns ``(line number, request)`` pairs and whether the input was a single object.
    """
    text = sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8")
    try:
        whole = json.loads(text)
    except json.JSONDecodeError:
        whole = None
    if isinstance(whole, dict):
        return [(1, whole)], True
    requests = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"safedecide decide: {source}:{n}: not JSON ({e.msg})") from e
        if not isinstance(request, dict):
            raise SystemExit(f"safedecide decide: {source}:{n}: expected a JSON object, got {type(request).__name__}")
        requests.append((n, request))
    if not requests:
        raise SystemExit(f"safedecide decide: {source} holds no requests")
    return requests, False


def cmd_decide(a):
    from .predict import SafeDecider

    requests, single = _read_requests(a.input)
    decider = SafeDecider.from_pretrained(a.model, revision=a.revision)
    try:
        rule = decider.rule(a.alpha)
    except ValueError as e:
        raise SystemExit(f"safedecide decide: {e}") from e
    if not rule.get("statement"):
        print(f"safedecide decide: {a.model} has no certified thresholds, so every answer escalates", file=sys.stderr)
    for n, request in requests:
        try:
            response = decider.decide(request, alpha=a.alpha)
        except (KeyError, ValueError, TypeError) as e:
            raise SystemExit(f"safedecide decide: {a.input}:{n}: {e}") from e
        print(json.dumps(response, ensure_ascii=False, indent=2 if single and not a.jsonl else None), flush=True)


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

    serve(a.model, a.host, a.port, revision=a.revision)


def cmd_mcp(a):
    from .mcp_server import serve

    serve(a.model, alpha=a.alpha, revision=a.revision)


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

    s = sub.add_parser(
        "calibrate",
        help="certify automate/escalate thresholds on labelled decisions",
        description="Certify automate/escalate thresholds on labelled decisions, for a SafeDecide checkpoint "
        "(--model; writes the thresholds next to its weights) or for any model behind a /v1/systemone "
        "endpoint (--http; writes a rule file).",
    )
    target = s.add_mutually_exclusive_group(required=True)
    target.add_argument("--model", help="local checkpoint or Hub repo id")
    target.add_argument("--http", help="URL of a /v1/systemone endpoint to certify (see README for the contract)")
    s.add_argument("--revision", help=REVISION_HELP)
    s.add_argument("--data", help="your labelled decisions, one JSON object per line (see README)")
    s.add_argument(
        "--output-dir",
        help="where the re-calibrated model is written: a Hub model is downloaded there, a local one copied",
    )
    s.add_argument("--in-place", action="store_true", help="local --model: replace the thresholds saved in it instead")
    s.add_argument(
        "--refit-temperatures",
        action="store_true",
        help="with --data: spend half of it refitting temperatures (default: keep the model's)",
    )
    s.add_argument(
        "--one-per-case",
        action="store_true",
        help="certify on one random decision per case (lines sharing a state or 'case'): valid for exchangeable cases",
    )
    s.add_argument("--alpha", type=float, default=0.1, help="default error level (5%%, 10%%, 20%% are all certified)")
    s.add_argument("--delta", type=float, default=0.1)
    s.add_argument("--max-rows", type=int, default=8000)
    s.add_argument("--seed", type=int, default=0)
    http = s.add_argument_group("with --http")
    http.add_argument("--output", default=DEFAULT_RULE_FILE, help="rule file to write (default: %(default)s)")
    http.add_argument("--http-model", help="'model' field to send in every request")
    http.add_argument("--header", action="append", help="request header 'Name: value' (repeatable); not saved")
    http.add_argument("--concurrency", type=int, default=4, help="requests in flight at once (default: %(default)s)")
    http.add_argument("--timeout", type=float, default=60.0, help="seconds per request (default: %(default)s)")
    http.add_argument(
        "--retries", type=int, default=3, help="retries on errors, timeouts, 429 and 5xx (default: %(default)s)"
    )
    s.set_defaults(fn=cmd_calibrate)

    s = sub.add_parser(
        "decide",
        help="answer /v1/systemone requests with automate/escalate and the guarantee (JSON out)",
        description="Answer one /v1/systemone request (a JSON object) or many (JSON Lines). Each result "
        "carries every answer's probabilities and its action (automate or escalate) and the guarantee "
        "behind 'automate'. A single request prints as indented JSON, several as one JSON object per line.",
    )
    s.add_argument("input", nargs="?", default="-", help="request file, JSON or JSONL (default: stdin)")
    s.add_argument("--model", required=True, help="local checkpoint or Hub repo id")
    s.add_argument("--revision", help=REVISION_HELP)
    s.add_argument("--alpha", type=float, help="certified error level to apply (default: the model's default)")
    s.add_argument("--jsonl", action="store_true", help="one compact JSON line per result, even for one request")
    s.set_defaults(fn=cmd_decide)

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
    s.add_argument("--revision", help=REVISION_HELP)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8177)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser(
        "mcp",
        help="serve decide and rule as MCP tools over stdio (needs safedecide[mcp])",
        description="An MCP server over stdio for agents (Claude Code, Cursor, ...): a 'decide' tool that "
        "answers typed questions with automate/escalate and the guarantee, and a read-only 'rule' tool "
        "that explains the certified rule. The error budget is fixed here, not by the agent.",
    )
    s.add_argument("--model", required=True, help="local checkpoint or Hub repo id")
    s.add_argument("--revision", help=REVISION_HELP)
    s.add_argument("--alpha", type=float, help="certified error level to apply (default: the model's default)")
    s.set_defaults(fn=cmd_mcp)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
