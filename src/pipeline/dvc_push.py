"""Save the current data/ as a new DVC version and upload it to the team bucket.

Usage: python -m src.pipeline.dvc_push

Runs as the last task of stoa_pipeline, after validate, so only data that passed the checks
becomes a version. It re-hashes data/raw, data/interim and data/processed (`dvc add`), which
rewrites their pointer files (data/*.dvc) if anything changed, then uploads the new files
(`dvc push`).

It does not commit the pointer files. Git history is what links a data version to the code
that made it, so a person commits them (`git add data/*.dvc && git commit`); the log lists
which ones changed. Until then, teammates' `dvc pull` still fetches the previous version.

Credentials: in Docker, GOOGLE_APPLICATION_CREDENTIALS points at a service account key in
.secrets/ (see README, "Run the pipeline"). Run by hand, it falls back to your own
`gcloud auth application-default login`. With neither (e.g. a fresh clone by someone outside
the team), it logs a warning and exits with SKIP_EXIT_CODE, which Airflow shows as skipped, so
the rest of the run still counts as a success.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from src.ingest.common import REPO_ROOT, setup_logging

TRACKED = ["data/raw", "data/interim", "data/processed"]

# @task.bash marks a task skipped (not failed) when its command exits with 99.
SKIP_EXIT_CODE = 99

log = logging.getLogger("stoa.pipeline")


class VersioningError(RuntimeError):
    """A dvc or git command failed, or credentials are missing."""


class MissingCredentials(VersioningError):
    """No key file and no gcloud login, so there is no way to reach the bucket."""


def gcloud_login_path() -> Path:
    """Where `gcloud auth application-default login` saves its credentials."""
    config = os.environ.get("CLOUDSDK_CONFIG", Path.home() / ".config" / "gcloud")
    return Path(config) / "application_default_credentials.json"


def check_credentials() -> None:
    """Fail early with a clear message if there are no credentials for the bucket."""
    key = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if key and not Path(key).is_file():
        raise MissingCredentials(
            f"GOOGLE_APPLICATION_CREDENTIALS points at {key}, which does not exist. "
            "Create a key for the stoa-airflow-dvc service account (README, 'Run the "
            "pipeline') or unset the variable to use your own gcloud login."
        )
    if not key and not gcloud_login_path().is_file():
        raise MissingCredentials(
            "no GOOGLE_APPLICATION_CREDENTIALS and no gcloud login. Run "
            "`gcloud auth application-default login` with an account on the project."
        )


def run(cmd: list[str], cwd: Path = REPO_ROOT) -> str:
    """Run cmd, log its output, and raise VersioningError with stderr if it fails."""
    log.info("running %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    for line in result.stdout.splitlines():
        log.info("  %s", line)
    if result.returncode != 0:
        raise VersioningError(
            f"{' '.join(cmd)} exited with {result.returncode}:\n{result.stderr.strip()}"
        )
    return result.stdout


def changed_pointers(cwd: Path = REPO_ROOT) -> list[str]:
    """Pointer files (data/*.dvc) that differ from the last git commit."""
    pointers = [f"{d}.dvc" for d in TRACKED]
    out = run(["git", "status", "--porcelain", "--", *pointers], cwd=cwd)
    return [line[3:] for line in out.splitlines() if line.strip()]


def version_data(cwd: Path = REPO_ROOT) -> list[str]:
    """dvc add + dvc push; return the pointer files that now need a git commit."""
    check_credentials()
    run(["dvc", "add", *TRACKED], cwd=cwd)
    run(["dvc", "push"], cwd=cwd)
    return changed_pointers(cwd)


def main() -> None:
    setup_logging()
    try:
        changed = version_data()
    except MissingCredentials as err:
        # Expected on a clone without access to the team bucket: the data is still built,
        # just not versioned, so skip rather than fail the run.
        log.warning(
            "skipping dvc_push, data not versioned (expected without access to the team "
            "bucket): %s",
            err,
        )
        sys.exit(SKIP_EXIT_CODE)
    if changed:
        log.info(
            "new data version pushed; commit the pointers to record it: git add %s && "
            'git commit -m "Data snapshot"',
            " ".join(changed),
        )
    else:
        log.info("data unchanged since the last committed version; nothing to commit")


if __name__ == "__main__":
    main()
