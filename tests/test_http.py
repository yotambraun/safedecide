"""SystemOneHTTPAdapter and `safedecide calibrate --http` against a local stub /v1/systemone server."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

httpx = pytest.importorskip("httpx")

from safedecide.adapters import SystemOneHTTPAdapter  # noqa: E402
from safedecide.cli import main  # noqa: E402
from safedecide.schema import Decision, labelled_decisions_from_jsonl  # noqa: E402


class Stub:
    """A /v1/systemone model: 'yes' when the state mentions an outage, with a fixed confidence per state."""

    def __init__(self, fail_first=0, delay=0.0):
        self.fail_first, self.delay = fail_first, delay
        self.bodies, self.active, self.peak, self.lock = [], 0, 0, threading.Lock()

    def answer(self, body):
        state = json.dumps(body["state"]) if not isinstance(body["state"], str) else body["state"]
        q = body["questions"]["q"]
        p_true = 0.97 if "outage" in state else 0.2 if "refund" in state else 0.55
        if q["type"] == "noul":
            return {"type": "noul", "noul": p_true, "probabilities": {"false": 1 - p_true, "true": p_true}}
        first, *rest = q["criteria"]
        probs = {o: (0.9 if o == first else 0.1 / len(rest)) for o in q["criteria"]}
        return {"type": "choice", "choice": first, "probabilities": probs}


@pytest.fixture
def server():
    started = []

    def start(stub):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with stub.lock:
                    stub.bodies.append({"path": self.path, "auth": self.headers.get("Authorization"), **body})
                    n = len(stub.bodies)
                    stub.active += 1
                    stub.peak = max(stub.peak, stub.active)
                try:
                    time.sleep(stub.delay)
                    if n <= stub.fail_first:
                        self.send_response(503)
                        self.send_header("Retry-After", "0")
                        self.end_headers()
                        return
                    data = json.dumps({"answers": {"q": stub.answer(body)}}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):  # the client timed out and left
                    pass
                finally:
                    with stub.lock:
                        stub.active -= 1

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        started.append(httpd)
        return f"http://127.0.0.1:{httpd.server_address[1]}"

    yield start
    for httpd in started:
        httpd.shutdown()
        httpd.server_close()


def _decisions(n):
    return [Decision(f"d{i}", "noul", f"state {i} outage", "urgent?", ["false", "true"], [0, 1]) for i in range(n)]


def test_adapter_keeps_order_and_bounds_concurrency(server):
    stub = Stub(delay=0.05)
    adapter = SystemOneHTTPAdapter(server(stub), max_concurrency=3, backoff=0)
    probs = adapter.predict_proba(_decisions(12))
    assert len(probs) == 12 and all(p[1] == pytest.approx(0.97) for p in probs)
    assert sorted(b["state"] for b in stub.bodies) == sorted(f"state {i} outage" for i in range(12))
    assert 1 < stub.peak <= 3
    assert all(b["path"] == "/v1/systemone" for b in stub.bodies)


def test_adapter_retries_transient_errors_then_gives_up(server):
    stub = Stub(fail_first=2)
    assert len(SystemOneHTTPAdapter(server(stub), retries=2, backoff=0).predict_proba(_decisions(1))) == 1
    assert len(stub.bodies) == 3  # two 503s, then the answer

    stub = Stub(fail_first=10)
    with pytest.raises(httpx.HTTPStatusError):
        SystemOneHTTPAdapter(server(stub), retries=1, backoff=0, max_concurrency=1).predict_proba(_decisions(1))
    assert len(stub.bodies) == 2


def test_adapter_times_out(server):
    stub = Stub(delay=0.5)
    adapter = SystemOneHTTPAdapter(server(stub), timeout=0.05, retries=1, backoff=0)
    with pytest.raises(httpx.TimeoutException):
        adapter.predict_proba(_decisions(1))
    assert len(stub.bodies) == 2  # retried once


def test_calibrate_http_certifies_the_endpoint_and_writes_a_rule_file(server, tmp_path, capsys):
    stub = Stub(fail_first=1)
    url = server(stub)
    rows = []
    for i in range(400):
        outage = i % 2 == 0
        rows.append(
            {
                "state": {"ticket": i, "text": "site outage" if outage else "refund please"},
                "question": {
                    "type": "noul",
                    "instructions": "Urgent?",
                    "criteria": {"false": "can wait", "true": "now"},
                },
                "answer": "yes" if outage else "no",
            }
        )
    data = tmp_path / "mine.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out = tmp_path / "rule.json"
    main(
        ["calibrate", "--http", url, "--data", str(data), "--output", str(out), "--concurrency", "4"]
        + ["--header", "Authorization: Bearer s3cret", "--http-model", "my-model"]
    )
    rule = json.loads(out.read_text(encoding="utf-8"))
    assert rule["endpoint"] == url + "/v1/systemone" and rule["model"] == "my-model"
    assert rule["selective_rule"]["alpha"] == 0.1
    assert [r["alpha"] for r in rule["selective_rules"]] == [0.05, 0.1, 0.2]
    assert rule["selective_rule"]["threshold"] is not None  # outage 0.97 and refund 0.8: all right, certifiable
    assert rule["calibration"]["n_selective"] == 400 and rule["calibration"]["n_cases"] == 400
    assert str(out) in capsys.readouterr().out

    sent = stub.bodies[-1]
    assert sent["auth"] == "Bearer s3cret" and sent["model"] == "my-model"
    assert isinstance(sent["state"], dict)  # the state exactly as in the file, not re-encoded as text
    assert sent["questions"]["q"]["criteria"] == {"false": "can wait", "true": "now"}


def test_calibrate_http_needs_data_and_rejects_checkpoint_options(tmp_path):
    with pytest.raises(SystemExit, match="--data"):
        main(["calibrate", "--http", "http://127.0.0.1:1"])
    data = tmp_path / "mine.jsonl"
    data.write_text(json.dumps({"state": "s", "question": {"type": "noul"}, "answer": True}), encoding="utf-8")
    with pytest.raises(SystemExit, match="--in-place"):
        main(["calibrate", "--http", "http://127.0.0.1:1", "--data", str(data), "--in-place"])


def test_labelled_lines_keep_the_request_as_written(tmp_path):
    row = {"state": {"a": 1}, "question": {"type": "noul", "instructions": "x?"}, "answer": True}
    path = tmp_path / "mine.jsonl"
    path.write_text(json.dumps(row), encoding="utf-8")
    (d,) = labelled_decisions_from_jsonl(str(path))
    assert d.meta["systemone"] == {"state": {"a": 1}, "question": {"type": "noul", "instructions": "x?"}}
