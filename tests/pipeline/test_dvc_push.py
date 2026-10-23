import subprocess

import pytest
from google.auth.exceptions import DefaultCredentialsError

from src.pipeline import dvc_push


@pytest.fixture
def repo(tmp_path):
    """A throwaway git repo holding the three committed pointer files and dvc.lock."""

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "data").mkdir()
    for d in dvc_push.TRACKED:
        (tmp_path / f"{d}.dvc").write_text("outs:\n- md5: aaa.dir\n")
    (tmp_path / dvc_push.LOCK).write_text("schema: '2.0'\n")
    git("add", ".")
    # gpgsign=false: machines that sign commits by default would otherwise need gpg here.
    git(
        "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
        "commit", "-qm", "pointers",
    )  # fmt: skip
    return tmp_path


def test_changed_pointers_lists_only_modified_pointer_files(repo):
    assert dvc_push.changed_pointers(repo) == []
    (repo / "data/raw.dvc").write_text("outs:\n- md5: bbb.dir\n")
    (repo / "notes.txt").write_text("not a pointer")
    assert dvc_push.changed_pointers(repo) == ["data/raw.dvc"]


def test_changed_pointers_include_dvc_lock(repo):
    (repo / dvc_push.LOCK).write_text("schema: '2.0'\nstages: {}\n")
    assert dvc_push.changed_pointers(repo) == [dvc_push.LOCK]


@pytest.fixture
def credentials(monkeypatch):
    """Stand in for google.auth.default; records GOOGLE_APPLICATION_CREDENTIALS per call."""
    seen = []

    def found():
        seen.append(dvc_push.os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
        return object(), "stoa"

    monkeypatch.setattr(dvc_push.google.auth, "default", found)
    monkeypatch.delenv("STOA_REQUIRE_DVC_PUSH", raising=False)
    return seen


@pytest.fixture
def no_credentials(monkeypatch):
    def missing():
        raise DefaultCredentialsError("none found")

    monkeypatch.setattr(dvc_push.google.auth, "default", missing)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("STOA_REQUIRE_DVC_PUSH", raising=False)


@pytest.fixture
def dvc_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(dvc_push, "run", lambda cmd, cwd=None: calls.append(cmd) or "")
    return calls


def test_version_data_adds_then_pushes(credentials, dvc_calls, tmp_path):
    dvc_push.version_data(tmp_path)
    assert dvc_calls[:3] == [
        ["dvc", "add", *dvc_push.TRACKED],
        [
            "dvc",
            "commit",
            "--force",
            "dvc.yaml",
        ],  # records each stage's inputs in dvc.lock
        ["dvc", "push"],
    ]


def test_no_credentials_fails_before_any_dvc_command(
    no_credentials, dvc_calls, tmp_path
):
    with pytest.raises(dvc_push.MissingCredentials, match="no gcloud login"):
        dvc_push.version_data(tmp_path)
    assert dvc_calls == []


def test_existing_key_file_is_used(credentials, monkeypatch, tmp_path):
    key = tmp_path / "key.json"
    key.write_text("{}")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(key))
    dvc_push.check_credentials()
    assert credentials == [str(key)]


def test_missing_key_file_falls_back_to_other_credentials(
    credentials, monkeypatch, tmp_path
):
    # On a Google Cloud VM there is no key file, but the VM's own account works.
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "missing.json"))
    dvc_push.check_credentials()
    assert credentials == [None]
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in dvc_push.os.environ


def test_main_skips_without_credentials(no_credentials, dvc_calls, caplog):
    with pytest.raises(SystemExit) as exit_info:
        dvc_push.main()
    assert exit_info.value.code == dvc_push.SKIP_EXIT_CODE
    assert "skipping dvc_push" in caplog.text
    assert dvc_calls == []


@pytest.mark.parametrize("value", ["1", "true", "YES"])
def test_main_fails_without_credentials_when_push_is_required(
    no_credentials, dvc_calls, monkeypatch, value
):
    monkeypatch.setenv("STOA_REQUIRE_DVC_PUSH", value)
    with pytest.raises(dvc_push.MissingCredentials):
        dvc_push.main()
    assert dvc_calls == []


def test_failed_command_raises_with_its_stderr(tmp_path):
    with pytest.raises(dvc_push.VersioningError, match="exited with 3:\nboom"):
        dvc_push.run(["sh", "-c", "echo boom >&2; exit 3"], cwd=tmp_path)
