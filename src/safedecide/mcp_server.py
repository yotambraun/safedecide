"""MCP server: SafeDecide's automate/escalate decisions as tools an agent can call.

    safedecide mcp --model yotambr/safedecide-140m [--alpha 0.1] [--revision REV]

Runs over stdio and needs the optional ``mcp`` package (``pip install "safedecide[mcp]"``).
Two tools, neither of which changes anything:

* ``decide``: answers one ``/v1/systemone`` request (a state and its typed questions) with
  the options' probabilities, each answer's ``action`` and the guarantee behind ``automate``.
* ``rule``: explains the certified rule the server applies.

The error budget is fixed when the server starts (``--alpha``, default: the model's default
level). An agent cannot loosen it per call, and there is no tool to re-certify: certifying
needs labelled data and rewrites the thresholds, which belongs with ``safedecide calibrate``,
not with the agent whose answers the rule is judging.
"""

from __future__ import annotations

from typing import Any

INSTRUCTIONS = (
    "SafeDecide answers typed decisions (noul = yes/no, choice = pick one option, score = a level 0..N) "
    "about a state, and marks each answer 'automate' or 'escalate'. Act on an answer alone only when its "
    "action is 'automate'; hand 'escalate' answers to a person or a stronger process. Call 'rule' to see "
    "the certified error budget behind 'automate'."
)


def describe_rule(config: dict[str, Any], alpha: float | None = None) -> dict[str, Any]:
    """The certified rule saved in a checkpoint's config, in a form an agent or a person can read.

    ``alpha`` picks the level in use (default: the config's default level). Raises
    ``ValueError`` when that level was never certified.
    """
    import math

    default = config.get("selective_rule") or {}
    rules = config.get("selective_rules") or ([default] if default else [])
    if alpha is None:
        in_use = default or None
    else:
        in_use = next((r for r in rules if r.get("alpha") is not None and math.isclose(r["alpha"], alpha)), None)
        if in_use is None:
            raise ValueError(f"no rule certified at alpha={alpha}; levels: {[r.get('alpha') for r in rules]}")
    levels = [
        {
            "alpha": r.get("alpha"),
            "delta": r.get("delta"),
            "threshold": r.get("threshold"),
            "certified": r.get("threshold") is not None,
            "calibration_automation": r.get("calibration_automation"),
            "n_calibration": r.get("n_calibration"),
        }
        for r in rules
    ]
    if in_use is None:
        explanation = "This model has no certified thresholds, so every answer is marked 'escalate'."
    elif in_use.get("threshold") is None:
        explanation = (
            f"At the error budget in use ({in_use['alpha']:.0%}) the calibration data could not certify any "
            "threshold, so every answer is marked 'escalate'."
        )
    else:
        explanation = (
            f"An answer is marked 'automate' when its confidence (the probability of its top option) is at least "
            f"{in_use['threshold']:.3f}, and 'escalate' otherwise. {in_use.get('statement') or ''}".strip()
        )
    return {
        "model": config.get("model_name") or config.get("run_name"),
        "in_use": None if in_use is None else {k: in_use.get(k) for k in ("alpha", "delta", "threshold")},
        "levels": levels,
        "calibration": config.get("calibration", {}),
        "explanation": explanation,
    }


def build_server(decider, alpha: float | None = None):
    """An ``MCPServer`` whose tools answer with ``decider`` (a :class:`~safedecide.SafeDecider`) at ``alpha``."""
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations

    describe_rule(decider.config, alpha)  # fail at start-up, not on the first call, if alpha was never certified
    server = MCPServer("safedecide", instructions=INSTRUCTIONS)
    read_only = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True)

    @server.tool(annotations=read_only)
    def decide(state: dict[str, Any] | str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Answer typed questions about one state, each marked 'automate' or 'escalate'.

        state: the facts of the case, as a JSON object or text (a ticket, an invoice, a trace...).
        questions: question id -> {"type", "instructions", "criteria"}. type "noul" is yes/no
        (criteria optional: {"false": ..., "true": ...}); "choice" picks one option (criteria:
        {option: description}, at least two); "score" picks a level (criteria: a list of level
        descriptions, at least two).

        Returns per question its probabilities, confidence and action, plus the guarantee
        behind every 'automate'. Act alone only on 'automate' answers.
        """
        try:
            return decider.decide({"state": state, "questions": questions}, alpha=alpha)
        except (KeyError, ValueError, TypeError) as e:  # a malformed request: tell the agent what to fix
            raise ToolError(f"invalid request: {e}") from e

    @server.tool(annotations=read_only)
    def rule() -> dict[str, Any]:
        """Explain the certified automate/escalate rule this server applies: the error budget, the
        threshold, every certified level and the calibration data behind them."""
        return describe_rule(decider.config, alpha)

    return server


def serve(model: str, alpha: float | None = None, revision: str | None = None) -> None:
    """Load ``model`` and serve it over stdio until the client disconnects."""
    try:
        import mcp  # noqa: F401
    except ImportError as e:
        raise SystemExit('safedecide mcp needs the mcp package: pip install "safedecide[mcp]"') from e
    from .predict import SafeDecider

    decider = SafeDecider.from_pretrained(model, revision=revision)
    try:
        server = build_server(decider, alpha)
    except ValueError as e:
        raise SystemExit(f"safedecide mcp: {e}") from e
    server.run("stdio")
