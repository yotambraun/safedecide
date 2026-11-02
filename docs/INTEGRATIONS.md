# Integrations

Three ways to put SafeDecide's automate/escalate rule next to code you already have:
the `decide` command, certifying a model you already serve over HTTP, and an MCP server
for agents.

## 1. `safedecide decide`

Answers `/v1/systemone` requests with a SafeDecide model, from a file or stdin:

```bash
safedecide decide --model yotambr/safedecide-140m request.json
cat requests.jsonl | safedecide decide --model yotambr/safedecide-140m --alpha 0.2 > results.jsonl
```

The input is one request (a JSON object) or JSON Lines of them, in the shape the README's
Quickstart sends to `decide`. Each result is the response `SafeDecider.decide` returns:
per question its probabilities, confidence and `action` (`automate` or `escalate`), plus the
`guarantee` (`alpha`, `delta` and the `statement` behind every `automate`). A single request
prints as indented JSON, several as one JSON object per line (`--jsonl` forces that for one
request too). `--alpha` picks one of the model's certified levels and `--revision` a Hub
revision. A malformed line stops the run with an error naming it.

## 2. Certify a model you already run

If your decision model sits behind an HTTP endpoint that speaks `/v1/systemone` (Jev, Clef,
d3, `safedecide serve`, or your own), SafeDecide can certify a threshold on that model's own
probabilities, without running a SafeDecide model:

```bash
safedecide calibrate --http https://decisions.internal:8000 --data my_decisions.jsonl \
  --output rule.json --header "Authorization: Bearer $DECISIONS_TOKEN"
```

`--data` is the labelled JSONL file described in the README (one question and its right
answer per line). Every line is sent once, the answers are scored against the labels, and
the thresholds are certified at 5%, 10% and 20% (`--alpha` sets the default, `--delta` the
confidence, `--one-per-case` the unit, as for `--model`).

### Request

One request per line of the file:

```
POST <URL>/v1/systemone          (appended unless the URL already ends with it)
Content-Type: application/json
<each --header>

{"state": <the line's "state", as written>,
 "questions": {"q": <the line's "question", as written>},
 "model": <--http-model, only when given>}
```

Up to `--concurrency` requests (default 4) are in flight at once; each has `--timeout`
seconds (default 60). Connection errors, timeouts and HTTP 408, 425, 429 and 5xx responses
are retried up to `--retries` times (default 3), waiting 0.5 s × 2^attempt with jitter, or
the server's `Retry-After` in seconds, at most 60 s. Any other error status stops the run.
Header values are sent but never written to the rule file.

### Response

A 2xx JSON body whose `answers.q` holds the answer, in the first of these forms present:

1. `probabilities`: an object with a probability per option. For `choice`, keyed by option
   name (the keys of `criteria`); for `score`, by level index as a string (`"0"`, `"1"`, ...);
   for `noul`, `"true"` and optionally `"false"` (or `"yes"`/`"no"`). A list in option order
   also works. Values are normalised to sum to 1.
2. For `noul`: `noul`, the probability of true.
3. `choice` (an option name) with `confidence`, its probability (1.0 if absent). The rest of
   the mass is spread evenly over the other options, which leaves the top answer and its
   confidence, all that the threshold uses, exact.

The confidence is the largest normalised probability, and an answer is correct when that
option is the line's answer (on a tie, the first option in order). Other fields are ignored.

### Rule file

Its shape, with illustrative values and two of the three levels:

```json
{
  "safedecide_version": "0.3.0",
  "endpoint": "https://decisions.internal:8000/v1/systemone",
  "model": null,
  "selective_rule": {"threshold": 0.83, "alpha": 0.1, "...": "the default level's entry from selective_rules"},
  "selective_rules": [
    {"threshold": null, "alpha": 0.05, "delta": 0.1, "n_calibration": 600,
     "calibration_automation": 0.0, "calibration_error_ucb": 1.0, "statement": "With probability ..."},
    {"threshold": 0.83, "alpha": 0.1, "delta": 0.1, "n_calibration": 600,
     "calibration_automation": 0.41, "calibration_error_ucb": 0.098, "statement": "With probability ..."}
  ],
  "calibration": {"n_selective": 600, "n_cases": 120, "one_per_case": false, "n_decisions": 600,
                  "accuracy": 0.81, "sources": {"jsonl": 600}}
}
```

`threshold` is `null` when nothing could be certified at that level: escalate everything.
`selective_rule` and `selective_rules` have the same shape as in a checkpoint's
`safedecide_config.json`. To apply the rule where you call the model:

```python
import json

rules = {r["alpha"]: r for r in json.load(open("rule.json"))["selective_rules"]}
threshold = rules[0.1]["threshold"]  # your error budget
automate = threshold is not None and confidence >= threshold  # confidence: the top probability, as above
```

The guarantee covers the endpoint as it was certified: the same model, prompt and settings,
on traffic exchangeable with the labelled file. Re-certify after changing any of them.
