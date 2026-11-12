# Airflow and MLflow on a cloud VM

The same `docker-compose.yaml` stack that runs on laptops (Airflow, its Postgres, MLflow) runs on one Compute Engine VM in the team project, so the daily pipeline and the shared experiment tracker don't depend on anyone's laptop being awake. Cloud Composer was ruled out on cost (CLAUDE.md, "Infrastructure"). Everything is in `ops/vm/`.

| | |
|---|---|
| VM | `stoa-ops`, project `stoa-511822`, zone `us-east1-b` (the bucket's region), `e2-standard-4` (4 vCPU, 16 GB), 50 GB disk, Debian 12 |
| Network | No public IP. SSH only from Identity-Aware Proxy (`stoa-allow-iap-ssh`, 35.235.240.0/20); outbound through Cloud NAT (`stoa-router`/`stoa-nat`); the bucket through Private Google Access |
| UIs | Bound to the VM's loopback (`ops/vm/compose.vm.yaml`); reached with an SSH tunnel |
| Repo | `/srv/stoa`, owned by the `stoa` user (uid 50000, the containers' user) |
| Credentials | The `stoa-airflow-dvc` key, as on laptops; the VM itself has no service account |
| MLflow artifacts | `gs://stoa-dvc-data/mlflow-artifacts` (outside DVC's paths); runs and the registry in the VM's Postgres |
| Uptime | On demand: starts at 23:40 UTC each day, and whenever someone runs `vm.sh`; stops itself when idle |

## Using it

You need gcloud logged in with your team account (all five of us are project Editors). From the repo root:

```bash
ops/vm/vm.sh tunnel     # starts the VM if needed; Airflow at http://localhost:8080, MLflow at http://localhost:5001
ops/vm/vm.sh status     # VM state, containers, last idle checks
ops/vm/vm.sh ssh        # a shell on the VM (repo at /srv/stoa; `sudo docker compose ...`)
ops/vm/vm.sh deploy     # pull main on the VM and restart the stack
ops/vm/vm.sh stop       # stop now instead of waiting for the idle stop
```

Stop your local stack first or use other ports: the tunnel needs local ports 8080 and 5001.

To log training runs from a laptop to the shared MLflow, keep a tunnel open and leave `MLFLOW_TRACKING_URI` at its default (`http://localhost:5001`, `configs/models.yaml`). Large training jobs are better run on the VM itself (`stoa_train` DAG, or `vm.sh ssh`), so the 7.4-million-row frame doesn't have to come down to a laptop. If a job needs more memory, `ops/vm/vm.sh resize e2-highmem-4` (32 GB) and back to `e2-standard-4` afterwards.

## When it runs and what it costs

- An instance schedule (`stoa-ops-daily-start`) starts the VM at 23:40 UTC (6:40 PM EST, 7:40 PM EDT), 20 minutes before `stoa_pipeline`'s 00:00 UTC run. `stoa_osm_watch` runs at 00:00 UTC on Sundays, so it is covered too.
- `ops/vm/idle-stop.sh` checks every 5 minutes and shuts the VM down after 3 idle checks in a row. Busy means: up under 30 minutes, a DAG run queued or running, or any SSH connection (which includes a tunnel). If Airflow's database doesn't answer, it waits up to 2 hours of uptime before stopping anyway.
- A normal day is about 45 minutes up (a weekly run about an hour). At about $0.13 per hour for `e2-standard-4` plus about $5 per month for the disk, that is roughly $10 per month, billed to the project's billing account.
- `stoa_minute_feed` stays paused on the VM: the Explorer cluster polls the feed (`ops/hpc/`), and a VM that slept between polls couldn't cover the 02:00 Melbourne reset.

## The VM as the team's pipeline runner

Once `stoa_pipeline` is unpaused on the VM, it is the one machine that runs it. Pause `stoa_pipeline` on laptop stacks, or every unpaused machine sends its own alert emails and pushes its own data version.

`dvc_push` on the VM leaves `data/*.dvc` and `dvc.lock` changed but uncommitted, as on laptops. To publish a version, copy those four files into a PR (`gcloud compute scp` through IAP, or `vm.sh ssh` and `sudo -u stoa git diff`). `vm.sh deploy` refuses during a DAG run, stops the scheduler while it pulls, and stops with an error if main changed those files too.

## Setting it up (done once)

1. Create the network pieces and the VM: `ops/vm/vm.sh create`.
2. Copy `.env` and the bucket key from your laptop, install Docker, clone the repo, pull the committed data and start the stack: `ops/vm/vm.sh setup` (takes about 15 minutes; safe to re-run, for example after changing `.env`). DAGs stay paused.
3. A project owner grants two roles, since Editors can't change IAM. The Editor role doesn't include IAP tunnelling, so each teammate needs `roles/iap.tunnelResourceAccessor` (without it, SSH fails with `4033: 'not authorized'`). And Compute Engine's service agent needs `roles/compute.instanceAdmin.v1` to start the VM on its schedule. Then anyone runs `ops/vm/vm.sh schedule`.

   ```bash
   for m in user:puneet.puri999@gmail.com user:arshmeharkaur05@gmail.com \
            user:shubham.hadawle@gmail.com user:bpaduru@gmail.com user:basak.r@northeastern.edu; do
     gcloud projects add-iam-policy-binding stoa-511822 --member "$m" \
       --role roles/iap.tunnelResourceAccessor --condition None
   done
   gcloud projects add-iam-policy-binding stoa-511822 \
     --member serviceAccount:service-993920967496@compute-system.iam.gserviceaccount.com \
     --role roles/compute.instanceAdmin.v1 --condition None
   ```

4. A project owner sets a budget alert on the project's billing account (Billing, Budgets & alerts), for example at $40 per month.
5. In the Airflow UI through the tunnel, unpause `stoa_pipeline` and `stoa_osm_watch`. Unpausing starts a run right away, which emails `STOA_ALERT_EMAILS` and pushes a data version.

To remove everything: `gcloud compute instances delete stoa-ops --zone us-east1-b`, then the resource policy, NAT, router and firewall rule.
