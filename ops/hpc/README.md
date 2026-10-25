# Per-minute feed poller on Northeastern's Explorer cluster

The portal's per-minute counts keep only the current Melbourne day, so they have to be polled before each Melbourne midnight. Laptops sleep; the cluster doesn't. Slurm's `scrontab` runs `poll_minute_feed.sh` every 15 minutes as a tiny job (one core, 1 GB, a few seconds) on the `sharing` CPU partition. That's the same `src.ingest.minute_feed --bucket` command the `stoa_minute_feed` DAG runs, so the cluster and anyone's Airflow can poll side by side: writes are checked against the day file's version, so none overwrites another.

## Access

The poller writes as the service account `stoa-minute-feed@stoa-511822.iam.gserviceaccount.com`. It holds `roles/storage.objectUser` on `gs://stoa-dvc-data` under an IAM condition limited to objects under `minute_feed/`, so it can't read or change the DVC data. Its key is `~/.config/stoa/minute-feed-sa.json` on the cluster, readable only by its owner, and exists nowhere else. To revoke it, delete the key or the account in the Google Cloud console (IAM, Service accounts).

## Set up (once)

From a login node:

```bash
git clone https://github.com/romit-basak/Stoa.git ~/stoa
cd ~/stoa
# Login nodes kill heavier processes, so build the venv in a short job:
srun --partition=sharing --time=00:20:00 --mem=2G -c 2 bash -c \
  "module load python/3.13.5 && python3 -m venv .venv-feed && \
   .venv-feed/bin/pip install -q -c requirements.txt pandas pyarrow requests pyyaml google-auth"
# Put the service account key at ~/.config/stoa/minute-feed-sa.json (chmod 600), then:
srun --partition=sharing --time=00:05:00 --mem=1G -c 1 ops/hpc/poll_minute_feed.sh
scrontab ops/hpc/scrontab.txt
```

## Check on it

```bash
scrontab -l                     # the schedule
squeue --me                     # a poll waiting or running
tail ~/stoa-minute-feed.log     # each poll's rows added
```

After changing the code on `main`: `cd ~/stoa && git pull`. The next poll uses it.
