"""Move Postgres backups between the ops VM and the bucket (used by backup.sh and restore.sh).

Runs inside the mlflow image, which has google-cloud-storage and the bucket key; the VM itself
has no gcloud and no service account. The shell scripts pass this file with `python -c`.

    put <stamp>/<file>    upload stdin
    get <stamp>/<file>    download to stdout
    list                  complete backups, newest first (one stamp per line)
    prune <days>          delete complete backups older than <days>, always keeping the newest
"""

import os
import sys
from datetime import UTC, datetime, timedelta

from google.cloud import storage

BUCKET = os.environ.get("STOA_BACKUP_BUCKET", "stoa-dvc-data")
PREFIX = os.environ.get("STOA_BACKUP_PREFIX", "backups/postgres").rstrip("/") + "/"
# Written last, so a stamp without it is a backup that didn't finish.
MARKER = "COMPLETE"
STAMP_FORMAT = "%Y%m%dT%H%M%SZ"


def complete_stamps(bucket):
    stamps = set()
    for blob in bucket.list_blobs(prefix=PREFIX):
        stamp, _, name = blob.name[len(PREFIX) :].partition("/")
        if name == MARKER:
            stamps.add(stamp)
    return sorted(stamps, reverse=True)


def main(argv):
    bucket = storage.Client().bucket(BUCKET)
    command = argv[0] if argv else ""
    if command == "put" and len(argv) == 2:
        # stdin is a pipe (not seekable); dumps are a few MB, so read it whole.
        bucket.blob(PREFIX + argv[1]).upload_from_string(
            sys.stdin.buffer.read(), timeout=600
        )
    elif command == "get" and len(argv) == 2:
        bucket.blob(PREFIX + argv[1]).download_to_file(sys.stdout.buffer, timeout=600)
    elif command == "list" and len(argv) == 1:
        for stamp in complete_stamps(bucket):
            print(stamp)
    elif command == "prune" and len(argv) == 2:
        cutoff = datetime.now(UTC) - timedelta(days=int(argv[1]))
        for stamp in complete_stamps(bucket)[1:]:
            taken = datetime.strptime(stamp, STAMP_FORMAT).replace(tzinfo=UTC)
            if taken < cutoff:
                blobs = list(bucket.list_blobs(prefix=f"{PREFIX}{stamp}/"))
                # The marker goes first, so an interrupted prune leaves an incomplete
                # backup (ignored by list) rather than a "complete" one with files missing.
                blobs.sort(key=lambda b: not b.name.endswith("/" + MARKER))
                for blob in blobs:
                    blob.delete()
                print(f"pruned {stamp}", file=sys.stderr)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
