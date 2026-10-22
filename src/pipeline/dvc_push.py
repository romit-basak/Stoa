"""Save the current data/ as a new DVC version and upload it to the team bucket.

Usage: python -m src.pipeline.dvc_push

Runs as the last task of stoa_pipeline, after validate, so only data that passed the checks
becomes a version. It re-hashes data/raw, data/interim and data/processed (`dvc add`), which
rewrites their pointer files (data/*.dvc) if anything changed, then uploads the new files
(`dvc push`).

It does not commit the pointer files. Git history is what links a data version to the code
that made it, so a person commits them (`git add data/*.dvc && git commit`); the log lists
which ones changed. Until then, teammates' `dvc pull` still fetches the previous version.

Credentials are found the way the Google client libraries find them (google.auth.default):
the service account key that GOOGLE_APPLICATION_CREDENTIALS points at (in Docker, a file in
.secrets/; see README, "Run the pipeline"), else your own `gcloud auth application-default
login`, else the service account of the Google Cloud VM it runs on. A configured key file
that doesn't exist is ignored, so a VM without one still uses its own account.

With no credentials at all (e.g. a fresh clone by someone outside the team), it logs a warning
and exits with SKIP_EXIT_CODE, which Airflow shows as skipped, so the rest of the run still
counts as a success. Team machines set STOA_REQUIRE_DVC_PUSH=1 in .env to fail instead, so a
lost key triggers the failure email rather than quietly leaving the data unversioned.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import google.auth
from google.auth.exceptions import DefaultCredentialsError

from src.ingest.common import REPO_ROOT, setup_logging

TRACKED = ["data/raw", "data/interim", "data/processed"]

# @task.bash marks a task skipped (not failed) when its command exits with 99.
SKIP_EXIT_CODE = 99

log = logging.getLogger("stoa.pipeline")


class VersioningError(RuntimeError):
    """A dvc or git command failed, or credentials are missing."""


class MissingCredentials(VersioningError):
    """No key file, gcloud login or VM account, so there is no way to reach the bucket."""


def check_credentials() -> None:
    """Fail early with a clear message if there are no credentials for the bucket."""
    key = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if key and not Path(key).is_file():
        # docker-compose.yaml always sets this path; without the file, google.auth and dvc
        # would stop there instead of trying a gcloud login or the VM's own account.
        log.info("no key file at %s, looking for other credentials", key)
        del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    try:
        google.auth.default()
    except DefaultCredentialsError as err:
        raise MissingCredentials(
            "no credentials for the DVC bucket: no service account key"
            + (f" at {key}" if key else "")
            + ", no gcloud login and no Google Cloud VM account. Create a key for the "
            "stoa-airflow-dvc service account (README, 'Run the pipeline') or run "
            "`gcloud auth application-default login`."
        ) from err


def require_push() -> bool:
    """True when STOA_REQUIRE_DVC_PUSH is set (team machines), so missing credentials fail."""
    value = os.environ.get("STOA_REQUIRE_DVC_PUSH", "")
    return value.strip().lower() in {"1", "true", "yes"}


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
        if require_push():
            raise
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
