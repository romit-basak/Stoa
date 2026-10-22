import subprocess

import pytest

from src.pipeline import dvc_push


@pytest.fixture
def repo(tmp_path):
    """A throwaway git repo holding the three committed pointer files."""

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "data").mkdir()
    for d in dvc_push.TRACKED:
        (tmp_path / f"{d}.dvc").write_text("outs:\n- md5: aaa.dir\n")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "pointers")
    return tmp_path


def test_changed_pointers_lists_only_modified_pointer_files(repo):
    assert dvc_push.changed_pointers(repo) == []
    (repo / "data/raw.dvc").write_text("outs:\n- md5: bbb.dir\n")
    (repo / "notes.txt").write_text("not a pointer")
    assert dvc_push.changed_pointers(repo) == ["data/raw.dvc"]


def test_version_data_adds_then_pushes(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(dvc_push, "run", lambda cmd, cwd=None: calls.append(cmd) or "")
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    dvc_push.version_data(tmp_path)
    assert calls[0] == ["dvc", "add", *dvc_push.TRACKED]
    assert calls[1] == ["dvc", "push"]


def test_missing_key_file_fails_before_any_dvc_command(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(dvc_push, "run", lambda cmd, cwd=None: calls.append(cmd) or "")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "missing.json"))
    with pytest.raises(dvc_push.VersioningError, match="does not exist"):
        dvc_push.version_data(tmp_path)
    assert calls == []


def test_existing_key_file_passes(monkeypatch, tmp_path):
    key = tmp_path / "key.json"
    key.write_text("{}")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(key))
    dvc_push.check_credentials()


def test_failed_command_raises_with_its_stderr(tmp_path):
    with pytest.raises(dvc_push.VersioningError, match="exited with 3:\nboom"):
        dvc_push.run(["sh", "-c", "echo boom >&2; exit 3"], cwd=tmp_path)
