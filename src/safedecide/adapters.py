"""Uniform interface over decision models, so any of them can be scored on the
same "automation at guaranteed error" metric.

An adapter returns, per decision, a probability vector aligned with
``decision.options``. When a model only exposes its top answer and a
confidence (e.g. Phocinae), the remaining mass is spread uniformly over the
other options: selective metrics and top-label ECE only use the top answer and
its confidence, so they stay exact; NLL/Brier for such models are approximate.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from .schema import Decision


class DecisionAdapter(Protocol):
    name: str

    def predict_proba(self, decisions: Sequence[Decision]) -> list[np.ndarray]: ...


def _from_top(k: int, top: int, conf: float) -> np.ndarray:
    conf = float(np.clip(conf, 0.0, 1.0))
    p = np.full(k, (1.0 - conf) / max(1, k - 1))
    p[top] = conf
    return p / p.sum()


class LocalAdapter:
    """A trained safedecide checkpoint."""

    def __init__(self, path: str, name: str | None = None, calibrated: bool = True, revision: str | None = None):
        from .predict import SafeDecider

        # full precision: as certified and as run on CPU
        self.decider = SafeDecider.from_pretrained(path, full_precision=True, revision=revision)
        self.calibrated = calibrated
        self.name = name or self.decider.config.get("run_name", path)

    def predict_proba(self, decisions):
        return self.decider.predict_proba(decisions, calibrated=self.calibrated)


def decision_to_systemone_question(d: Decision) -> dict[str, Any]:
    if d.kind == "noul":
        q: dict[str, Any] = {"type": "noul", "instructions": d.question}
    elif d.kind == "score":
        crit = [d.option_text(i) for i in range(d.num_options)]
        q = {"type": "score", "instructions": d.question, "criteria": crit}
    else:
        descs = d.option_descriptions or [None] * d.num_options
        q = {"type": "choice", "instructions": d.question, "criteria": dict(zip(d.options, descs))}
    return q


def parse_systemone_answer(d: Decision, ans: dict[str, Any]) -> np.ndarray:
    """Tolerant parser for the response shapes of Jev, Clef, d3, jevos, ollajev."""
    probs = ans.get("probabilities")
    k = d.num_options
    if isinstance(probs, dict) and probs:
        if d.kind == "noul":
            pt = probs.get("true", probs.get("yes"))
            pf = probs.get("false", probs.get("no"))
            if pt is not None:
                pf = 1.0 - float(pt) if pf is None else float(pf)
                p = np.array([pf, float(pt)])
                return p / p.sum()
        else:
            keyed = [probs.get(o, probs.get(str(i))) for i, o in enumerate(d.options)]
            if all(v is not None for v in keyed):
                p = np.array(keyed, dtype=float)
                return p / p.sum()
    if isinstance(probs, list) and len(probs) == k:
        p = np.array(probs, dtype=float)
        return p / p.sum()
    if d.kind == "noul" and "noul" in ans:
        pt = float(ans["noul"])
        return np.array([1.0 - pt, pt])
    conf = float(ans.get("confidence", 1.0))
    choice = ans.get("choice")
    if choice in d.options:
        return _from_top(k, d.options.index(choice), conf)
    raise ValueError(f"cannot parse answer for {d.id}: {json.dumps(ans)[:200]}")


# Worth another try: request timeout, too early, rate limited, and the 5xx a restarting server returns.
RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class SystemOneHTTPAdapter:
    """Any server speaking ``POST /v1/systemone`` (Jev, Clef, d3/vLLM, jevos, ollajev, ``safedecide serve``...).

    One request per decision, ``{"state": ..., "questions": {"q": {...}}}`` (plus ``"model"``
    when given), sent by up to ``max_concurrency`` threads at a time. A decision read from
    labelled JSONL is sent with its ``state`` and ``question`` exactly as written there.
    The answer is read from ``answers.q`` by :func:`parse_systemone_answer`.

    ``timeout`` is in seconds per request. Connection errors, timeouts and HTTP 408, 425,
    429 and 5xx responses are retried up to ``retries`` times, waiting ``backoff * 2**attempt``
    seconds (with jitter, or the server's ``Retry-After``, at most 60 s) in between; any other
    HTTP error is raised at once. Results come back in input order.
    """

    def __init__(
        self,
        url: str,
        model: str | None = None,
        name: str | None = None,
        headers: dict | None = None,
        timeout: float = 60.0,
        max_concurrency: int = 4,
        retries: int = 3,
        backoff: float = 0.5,
    ):
        if max_concurrency < 1 or retries < 0 or backoff < 0 or timeout <= 0:
            raise ValueError("need max_concurrency >= 1, retries >= 0, backoff >= 0 and timeout > 0")
        self.url = url.rstrip("/")
        if not self.url.endswith("/v1/systemone"):
            self.url += "/v1/systemone"
        self.model = model
        self.name = name or model or url
        self.headers = headers or {}
        self.timeout = timeout
        self.max_concurrency = max_concurrency
        self.retries = retries
        self.backoff = backoff

    def request_body(self, d: Decision) -> dict[str, Any]:
        written = d.meta.get("systemone")
        if written:
            body: dict[str, Any] = {"state": written["state"], "questions": {"q": written["question"]}}
        else:
            body = {"state": d.state, "questions": {"q": decision_to_systemone_question(d)}}
        if self.model:
            body["model"] = self.model
        return body

    def _wait(self, attempt: int, response=None) -> None:
        delay = self.backoff * 2**attempt * random.uniform(0.5, 1.0)
        retry_after = response.headers.get("Retry-After") if response is not None else None
        if retry_after is not None:
            try:
                delay = float(retry_after)
            except ValueError:  # an HTTP date: keep our own backoff
                pass
        time.sleep(min(max(delay, 0.0), 60.0))

    def _post(self, client, body: dict[str, Any]) -> dict[str, Any]:
        import httpx

        for attempt in range(self.retries + 1):
            last = attempt == self.retries
            try:
                r = client.post(self.url, json=body)
            except httpx.TransportError:  # connection errors and timeouts
                if last:
                    raise
                self._wait(attempt)
                continue
            if r.status_code in RETRY_STATUS and not last:
                self._wait(attempt, r)
                continue
            r.raise_for_status()
            return r.json()
        raise AssertionError("unreachable")

    def predict_proba(self, decisions):
        from concurrent.futures import ThreadPoolExecutor

        import httpx

        limits = httpx.Limits(max_connections=self.max_concurrency)
        with httpx.Client(timeout=self.timeout, headers=self.headers, limits=limits) as client:

            def one(d: Decision) -> np.ndarray:
                data = self._post(client, self.request_body(d))
                try:
                    answer = data["answers"]["q"]
                except (KeyError, TypeError) as e:
                    raise ValueError(f"{self.url}: the response for {d.id} has no answers.q") from e
                return parse_systemone_answer(d, answer)

            pool = ThreadPoolExecutor(max_workers=self.max_concurrency)
            try:
                return list(pool.map(one, decisions))
            finally:  # on an error, drop the requests not sent yet
                pool.shutdown(wait=True, cancel_futures=True)


class PhocinaeAdapter:
    """Phocinae-Largha-150M-v1 through the ``phocinae-server`` package (>= 0.1.7).

    Use :meth:`local` for benchmarking. It calls the engine in-process, as the
    model's own typed-decisions harness does, so every question keeps its
    ``instructions`` and ``criteria``. The HTTP server's request schema (0.1.7)
    drops those two fields, and the model card measures 0.514 instead of 0.9055
    accuracy without them, so HTTP numbers are not this model's level.

    One call per case with all of its questions; probabilities come from the
    per-option ``option_scores``. Answers are bool (noul), 0-based index (choice)
    or ``2 + level`` (score).
    """

    name = "Phocinae-Largha-150M-v1"

    def __init__(self, url: str = "http://127.0.0.1:8155", timeout: float = 120.0, engine=None):
        self.url = url.rstrip("/") + "/v1/systemone"
        self.timeout = timeout
        self.engine = engine

    @classmethod
    def local(cls, model_dir: str, device: str = "auto") -> PhocinaeAdapter:
        """In-process engine on a local copy of the model repo (``snapshot_download``)."""
        from phocinae.engine import Engine

        return cls(engine=Engine(model_dir, device=device, model_name=cls.name, warm=False))

    def _run(self, client, state: str, questions: list[dict[str, Any]]) -> dict[str, Any]:
        if self.engine is not None:
            answers, confs, _actions, _usage, scores = self.engine.run(state, questions, with_scores=True)
            return {"answers": answers, "answer_confidence": confs, "option_scores": scores}
        r = client.post(self.url, json={"model": self.name, "state": state, "questions": questions})
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _question(qid: str, d: Decision) -> dict[str, Any]:
        q: dict[str, Any] = {"id": qid, "type": d.kind, "instructions": d.question}
        descs = d.option_descriptions or [None] * d.num_options
        if d.kind == "choice":
            q["options"] = list(d.options)
            q["criteria"] = {o: desc for o, desc in zip(d.options, descs) if desc}
        elif d.kind == "score":
            q["criteria"] = [desc or o for o, desc in zip(d.options, descs)]
        elif any(descs):
            q["criteria"] = dict(zip(("false", "true"), descs))
        return q

    @staticmethod
    def _parse(d: Decision, data: dict[str, Any], qid: str) -> np.ndarray:
        scores = (data.get("option_scores") or {}).get(qid)
        if scores is not None and len(scores) == d.num_options:
            p = np.clip(np.asarray(scores, dtype=float), 1e-9, None)
            return p / p.sum()
        ans = data["answers"][qid]
        conf = float((data.get("answer_confidence") or {}).get(qid, 1.0))
        top = {"noul": lambda a: int(bool(a)), "score": lambda a: int(a) - 2}.get(d.kind, int)(ans)
        return _from_top(d.num_options, top, conf)

    def predict_proba(self, decisions):
        import contextlib

        if self.engine is None:
            import httpx

            client_ctx = httpx.Client(timeout=self.timeout)
        else:
            client_ctx = contextlib.nullcontext()
        cases: dict[str, list[int]] = {}
        for i, d in enumerate(decisions):
            cases.setdefault(d.state, []).append(i)
        out: list[np.ndarray | None] = [None] * len(decisions)
        with client_ctx as client:
            for state, idx in cases.items():
                for start in range(0, len(idx), 64):  # server limit: 64 questions per request
                    chunk = idx[start : start + 64]
                    questions = [self._question(f"q{j}", decisions[i]) for j, i in enumerate(chunk)]
                    data = self._run(client, state, questions)
                    for j, i in enumerate(chunk):
                        out[i] = self._parse(decisions[i], data, f"q{j}")
        return out
