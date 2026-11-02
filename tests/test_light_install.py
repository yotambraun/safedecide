"""The statistics API must work where torch and transformers cannot be imported."""

import subprocess
import sys
import textwrap

BLOCKER = """
import importlib.abc
import sys


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("torch", "transformers"):
            raise ImportError(f"{name} is blocked in this test")
        return None


sys.meta_path.insert(0, Block())
"""


def run_without_torch(code: str) -> subprocess.CompletedProcess:
    script = BLOCKER + textwrap.dedent(code)
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)


def test_statistics_api_imports_and_runs_without_torch():
    result = run_without_torch(
        """
        import numpy as np

        import safedecide
        from safedecide import Decision, calibrate_selective, decisions_from_systemone_request, one_per_case
        from safedecide.calibration import calibrated_probs, fit_question_bias, fit_temperatures
        from safedecide.pipeline import certify_rules

        rng = np.random.default_rng(0)
        conf = rng.uniform(0.5, 1.0, 2000)
        correct = rng.uniform(0, 1, 2000) < conf
        assert np.isfinite(safedecide.calibrate_selective(conf, correct, alpha=0.2, delta=0.1).threshold)
        assert len(one_per_case([0, 0, 1])) == 2
        rules, _ = certify_rules(conf, correct, list(range(2000)), 0.1, 0.1)
        assert len(rules) == 3

        logits = [rng.normal(size=3) * 4 for _ in range(200)]
        targets = [np.eye(3)[int(z.argmax())] for z in logits]
        temps = fit_temperatures(logits, targets, ["choice"] * 200)
        assert temps["choice"] < 1.0  # confident and always right: sharpen
        d = Decision("x", "choice", "s", "q", ["a", "b", "c"], [1, 0, 0])
        assert fit_question_bias(logits, [d] * 200, temps) != {}
        assert abs(calibrated_probs(logits[:1], [d], temps)[0].sum() - 1) < 1e-9
        assert decisions_from_systemone_request({"state": "s", "questions": {"q": {"type": "noul"}}})[0].kind == "noul"

        import sys
        assert "torch" not in sys.modules and "transformers" not in sys.modules
        try:
            safedecide.SafeDecider
        except ImportError as e:
            assert "torch" in str(e)
        else:
            raise AssertionError("SafeDecider needs torch")
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_safedecider_is_still_importable_from_the_package():
    import safedecide
    from safedecide.predict import SafeDecider

    assert safedecide.SafeDecider is SafeDecider
    assert "SafeDecider" in dir(safedecide)
