"""The MCP server (`safedecide mcp`): skipped unless the optional `mcp` package is installed."""

import pytest

from safedecide.mcp_server import describe_rule

RULES = [
    {"alpha": 0.05, "delta": 0.1, "threshold": None, "n_calibration": 120, "calibration_automation": 0.0},
    {
        "alpha": 0.1,
        "delta": 0.1,
        "threshold": 0.82,
        "n_calibration": 120,
        "calibration_automation": 0.4,
        "statement": "the promise",
    },
]
CONFIG = {
    "model_name": "tiny",
    "selective_rule": RULES[1],
    "selective_rules": RULES,
    "calibration": {"n_selective": 120, "n_cases": 120, "one_per_case": True},
}


def test_describe_rule_explains_the_level_in_use():
    out = describe_rule(CONFIG)
    assert out["in_use"]["alpha"] == 0.1 and out["in_use"]["threshold"] == 0.82
    assert [lv["certified"] for lv in out["levels"]] == [False, True]
    assert out["calibration"]["one_per_case"]
    assert "0.82" in out["explanation"] and "the promise" in out["explanation"]
    stricter = describe_rule(CONFIG, alpha=0.05)
    assert stricter["in_use"]["threshold"] is None and "escalate" in stricter["explanation"]
    with pytest.raises(ValueError, match="no rule certified"):
        describe_rule(CONFIG, alpha=0.3)


def test_describe_rule_on_an_uncalibrated_model():
    out = describe_rule({})
    assert out["in_use"] is None and out["levels"] == []
    assert "no certified" in out["explanation"]


class FakeDecider:
    """Stands in for SafeDecider: the server only needs `config` and `decide`."""

    config = CONFIG

    def __init__(self):
        self.calls = []

    def decide(self, request, alpha=None):
        self.calls.append((request, alpha))
        if request["questions"].get("bad"):
            raise ValueError("choice question bad needs a dict of >=2 criteria")
        return {"answers": {"q": {"action": "automate", "confidence": 0.9}}, "guarantee": {"alpha": alpha}}


def _call(server, tool, arguments):
    import asyncio

    from mcp import Client

    async def run():
        async with Client(server) as client:
            tools = {t.name: t for t in (await client.list_tools()).tools}
            return tools, await client.call_tool(tool, arguments)

    return asyncio.run(run())


def test_server_tools_answer_and_explain():
    pytest.importorskip("mcp")
    from safedecide.mcp_server import build_server

    decider = FakeDecider()
    server = build_server(decider, alpha=0.1)
    question = {"type": "noul", "instructions": "urgent?"}
    tools, result = _call(server, "decide", {"state": {"ticket": "outage"}, "questions": {"q": question}})
    assert set(tools) == {"decide", "rule"}  # no tool can re-certify or change the budget
    assert all(t.annotations.read_only_hint for t in tools.values())
    assert "alpha" not in tools["decide"].input_schema["properties"]
    assert not result.is_error
    assert result.structured_content["answers"]["q"]["action"] == "automate"
    assert decider.calls == [({"state": {"ticket": "outage"}, "questions": {"q": question}}, 0.1)]

    _, result = _call(server, "rule", {})
    assert result.structured_content["in_use"] == {"alpha": 0.1, "delta": 0.1, "threshold": 0.82}

    _, result = _call(server, "decide", {"state": "x", "questions": {"bad": {"type": "choice"}}})
    assert result.is_error and "criteria" in result.content[0].text


def test_server_refuses_a_budget_that_was_never_certified():
    pytest.importorskip("mcp")
    from safedecide.mcp_server import build_server

    with pytest.raises(ValueError, match="no rule certified"):
        build_server(FakeDecider(), alpha=0.3)


@pytest.mark.e2e
def test_mcp_command_over_stdio(tmp_path):
    """`safedecide mcp` as Claude Code or Cursor would launch it: a subprocess speaking MCP on stdio."""
    pytest.importorskip("mcp")
    pytest.importorskip("torch")
    import asyncio
    import json
    import sys

    from mcp import Client, StdioServerParameters
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "outage", "yes", "no", "urgent", "?", "[", "]", "noul"]
    (tmp_path / "vocab.txt").write_text("\n".join(words), encoding="utf-8")
    BertTokenizer(str(tmp_path / "vocab.txt")).save_pretrained(tmp_path)
    cfg = BertConfig(vocab_size=len(words), hidden_size=16, num_hidden_layers=1, num_attention_heads=1)
    BertForSequenceClassification(BertConfig(**{**cfg.to_dict(), "num_labels": 1})).save_pretrained(tmp_path)
    (tmp_path / "safedecide_config.json").write_text(json.dumps(CONFIG), encoding="utf-8")

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "safedecide.cli", "mcp", "--model", str(tmp_path)]
    )

    async def run():
        async with Client(params) as client:
            question = {"type": "noul", "instructions": "urgent?"}
            decided = await client.call_tool("decide", {"state": "outage", "questions": {"u": question}})
            explained = await client.call_tool("rule", {})
            return decided, explained

    decided, explained = asyncio.run(run())
    answer = decided.structured_content["answers"]["u"]
    assert answer["action"] == ("automate" if answer["confidence"] >= 0.82 else "escalate")
    assert decided.structured_content["guarantee"]["statement"] == "the promise"
    assert explained.structured_content["in_use"]["threshold"] == 0.82
