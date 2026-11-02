"""SafeDecider loading, without a model: the Hub download and the backbone are faked."""

import json

import huggingface_hub
import pytest

pytest.importorskip("torch")

from safedecide import predict  # noqa: E402
from safedecide.predict import CONFIG_NAME, SafeDecider, resolve_model_dir  # noqa: E402


class FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    """A Hub whose repo has a `main` and a `v0.2` revision with different certified thresholds."""
    calls = []

    def snapshot_download(repo_id, revision=None, local_dir=None, **kwargs):
        calls.append({"repo_id": repo_id, "revision": revision, "local_dir": local_dir})
        folder = tmp_path / "hub" / (revision or "main")
        folder.mkdir(parents=True, exist_ok=True)
        rule = {"threshold": 0.7 if revision == "v0.2" else 0.9, "alpha": 0.1}
        (folder / CONFIG_NAME).write_text(json.dumps({"selective_rule": rule}), encoding="utf-8")
        return str(folder)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(predict, "load_backbone", lambda path: (object(), FakeModel()))
    return calls


def test_from_pretrained_passes_revision_to_the_hub(fake_hub):
    assert SafeDecider.from_pretrained("me/model", device="cpu").threshold == 0.9
    pinned = SafeDecider.from_pretrained("me/model", device="cpu", revision="v0.2")
    assert pinned.threshold == 0.7
    assert [c["revision"] for c in fake_hub] == [None, "v0.2"]


def test_resolve_model_dir_passes_revision_and_local_dir(fake_hub, tmp_path):
    resolve_model_dir("me/model", local_dir=str(tmp_path / "copy"), revision="abc123")
    assert fake_hub[-1] == {"repo_id": "me/model", "revision": "abc123", "local_dir": str(tmp_path / "copy")}


def test_revision_on_a_local_folder_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="revision"):
        resolve_model_dir(str(tmp_path), revision="v0.2")
