"""Copy the latest flow-model results out of MLflow into the docs, as charts and a summary.

Usage:
  python -m src.models.flow.figures [--out docs/models/figures] [--variant rich|portable|both]

Needs the MLflow server (the VM's through `ops/vm/vm.sh tunnel`, or MLFLOW_TRACKING_URI). For
each variant, it takes the latest selected run (stoa.role = selected) and the runs of the same
training: the tuning searches started before it and the bias mitigations started after it, on
the same data version. It writes to --out:

- model_comparison_<variant>.png, calibration_curve_<variant>.png,
  feature_importance_<variant>.png: the charts select.py logged with the selected run
  (model ranking and bias check, the calibration curve, SHAP importance);
- mitigation_<variant>.png: WAPE and worst slice under each bias mitigation, rebuilt from the
  mitigation runs' metrics;
- hyperparameter_importance_<variant>.png and tuning_history_<variant>.png: Optuna's estimate
  of each hyperparameter's importance per family, and each family's trials with the best so
  far (hyperparameter sensitivity), when the training was tuned;
- summary.json: the run ids, metrics and tuned values behind the charts, so the docs' numbers
  can be traced to MLflow.

Charts missing from a run (an untuned training has no tuning charts) are skipped and listed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.ingest.common import REPO_ROOT, setup_logging
from src.models import tracking
from src.models.flow import data as flow_data
from src.models.flow import mitigate

SELECTION_CHARTS = ("model_comparison", "calibration_curve", "feature_importance")
SUMMARY_METRICS = (
    "log_mae_cv",
    "wape_cv",
    "coverage80_cv",
    "sensor_rank_corr_cv",
    "wape_holdout",
    "log_mae_holdout",
    "worst_slice_ratio",
)


def _search(client: MlflowClient, exp_id: str, filters: list[str], **kw) -> list[Any]:
    return client.search_runs(
        [exp_id],
        filter_string=" and ".join(["attributes.status = 'FINISHED'", *filters]),
        order_by=["attributes.start_time DESC"],
        **kw,
    )


def selected_run(client: MlflowClient, exp_id: str, variant: str) -> Any | None:
    runs = _search(
        client,
        exp_id,
        ["tags.`stoa.role` = 'selected'", f"tags.`stoa.variant` = '{variant}'"],
        max_results=1,
    )
    return runs[0] if runs else None


def mitigation_runs(client: MlflowClient, exp_id: str, sel: Any) -> list[Any]:
    """The latest run of each mitigation for the selected run's variant and data version,
    started after it (mitigate runs after select)."""
    tags = sel.data.tags
    runs = _search(
        client,
        exp_id,
        [
            "tags.`stoa.role` = 'mitigation'",
            f"tags.`stoa.variant` = '{tags.get('stoa.variant', '')}'",
            f"tags.`stoa.dvc.processed` = '{tags.get('stoa.dvc.processed', '')}'",
            f"attributes.start_time >= {sel.info.start_time}",
        ],
        max_results=100,
    )
    latest: dict[str, Any] = {}
    for r in reversed(runs):  # oldest first, so the latest of each name wins
        latest[r.data.tags.get("stoa.mitigation", r.info.run_name)] = r
    return list(latest.values())


def tuning_runs(
    client: MlflowClient, exp_id: str, sel: Any, window_ms: int = 24 * 3600 * 1000
) -> dict[str, Any]:
    """The latest tuning search per family for the selected run's variant, from the same
    training: the same code and data (git commit and DVC version), started before it and
    within `window_ms` of it. An untuned training that follows a tuned one on other code
    gets no tuning charts, rather than the earlier search's."""
    tags = sel.data.tags
    start = sel.info.start_time
    filters = [
        "tags.`stoa.role` = 'tuning'",
        f"tags.`stoa.variant` = '{tags.get('stoa.variant', '')}'",
        f"attributes.start_time < {start}",
        f"attributes.start_time > {start - window_ms}",
    ]
    for key in ("stoa.git_commit", "stoa.dvc.processed"):
        if tags.get(key):
            filters.append(f"tags.`{key}` = '{tags[key]}'")
    runs = _search(client, exp_id, filters, max_results=50)
    found: dict[str, Any] = {}
    for r in runs:  # newest first
        found.setdefault(r.data.tags.get("stoa.family", r.info.run_name), r)
    return found


def trials(client: MlflowClient, exp_id: str, parent: Any) -> pd.DataFrame:
    """One row per Optuna trial of a tuning search: number and inner-fold log MAE."""
    runs = _search(
        client,
        exp_id,
        [f"tags.`mlflow.parentRunId` = '{parent.info.run_id}'"],
        max_results=1000,
    )
    rows = [
        {
            "trial": int(r.info.run_name.removeprefix("trial_")),
            "log_mae_inner": r.data.metrics.get("log_mae_inner"),
        }
        for r in runs
        if r.info.run_name.startswith("trial_")
    ]
    if not rows:
        return pd.DataFrame(columns=["trial", "log_mae_inner"])
    return pd.DataFrame(rows).sort_values("trial").reset_index(drop=True)


def fetch(run_id: str, path: str, tmp: str) -> Path | None:
    """One artifact of a run, downloaded under `tmp`; None if the run doesn't have it."""
    try:
        return Path(
            mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=path, dst_path=tmp
            )
        )
    # A missing artifact is an MlflowException from the server, an OSError from a file store.
    except (MlflowException, OSError):
        return None


def download(run_id: str, path: str, dest: Path) -> bool:
    """Copy one artifact to `dest`; False if the run doesn't have it."""
    with tempfile.TemporaryDirectory() as tmp:
        local = fetch(run_id, path, tmp)
        if local is None:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, dest)
    return True


def load_json_artifact(run_id: str, path: str) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory() as tmp:
        local = fetch(run_id, path, tmp)
        return json.loads(local.read_text()) if local else None


def importance_chart(importances: dict[str, dict[str, float]], path: Path) -> None:
    """One panel per family: Optuna's importance of each hyperparameter."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fams = sorted(importances)
    fig, axes = plt.subplots(
        1, len(fams), figsize=(4.2 * len(fams), 3.6), squeeze=False
    )
    for ax, fam in zip(axes[0], fams, strict=True):
        imp = pd.Series(importances[fam]).sort_values()
        ax.barh(imp.index, imp.to_numpy(), color="#2e86c1")
        ax.set_title(fam)
        ax.set_xlabel("relative importance for log MAE")
    fig.suptitle("Hyperparameter importance (Optuna, from the tuning trials)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def history_chart(histories: dict[str, pd.DataFrame], path: Path) -> None:
    """Each family's trials and the best so far, on the inner-fold log MAE."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for fam in sorted(histories):
        h = histories[fam].dropna()
        if h.empty:
            continue
        (line,) = ax.plot(
            h["trial"], h["log_mae_inner"].cummin(), label=f"{fam} best so far"
        )
        ax.scatter(
            h["trial"], h["log_mae_inner"], s=10, alpha=0.4, color=line.get_color()
        )
    ax.set_xlabel("Optuna trial")
    ax.set_ylabel("log MAE on the inner location folds")
    ax.set_title("Tuning: every trial (dots) and the best so far (lines)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def export_variant(
    client: MlflowClient, exp_id: str, variant: str, out: Path
) -> dict[str, Any]:
    sel = selected_run(client, exp_id, variant)
    if sel is None:
        return {"variant": variant, "error": "no selected run"}
    out.mkdir(parents=True, exist_ok=True)
    written, missing = [], []
    for name in SELECTION_CHARTS:
        dest = out / f"{name}_{variant}.png"
        if download(sel.info.run_id, f"selection/{name}.png", dest):
            written.append(dest.name)
        else:
            missing.append(dest.name)

    mit = mitigation_runs(client, exp_id, sel)
    mitigations = [
        {
            "mitigation": r.data.tags.get("stoa.mitigation", r.info.run_name),
            "run_id": r.info.run_id,
            "wape_cv": r.data.metrics.get("wape_cv"),
            "log_mae_cv": r.data.metrics.get("log_mae_cv"),
            "worst_slice_ratio": r.data.metrics.get("worst_slice_ratio"),
        }
        for r in mit
    ]
    if mitigations:
        mitigate.chart(pd.DataFrame(mitigations), out / f"mitigation_{variant}.png")
        written.append(f"mitigation_{variant}.png")
    else:
        missing.append(f"mitigation_{variant}.png")

    tuned = tuning_runs(client, exp_id, sel)
    importances, histories, best = {}, {}, {}
    for fam, run in tuned.items():
        imp = load_json_artifact(run.info.run_id, "param_importance.json")
        if imp:
            importances[fam] = imp
        histories[fam] = trials(client, exp_id, run)
        best[fam] = {
            "run_id": run.info.run_id,
            "log_mae_inner_best": run.data.metrics.get("log_mae_inner_best"),
            "params": {
                k.removeprefix("best_"): v
                for k, v in run.data.params.items()
                if k.startswith("best_")
            },
            "trials": len(histories[fam]),
        }
    if importances:
        importance_chart(importances, out / f"hyperparameter_importance_{variant}.png")
        written.append(f"hyperparameter_importance_{variant}.png")
    else:
        missing.append(f"hyperparameter_importance_{variant}.png")
    if any(not h.empty for h in histories.values()):
        history_chart(histories, out / f"tuning_history_{variant}.png")
        written.append(f"tuning_history_{variant}.png")
    else:
        missing.append(f"tuning_history_{variant}.png")

    m = sel.data.metrics
    return {
        "variant": variant,
        "selected_run_id": sel.info.run_id,
        "family": sel.data.params.get("family"),
        "data_version": sel.data.tags.get("stoa.dvc.processed"),
        "git_commit": sel.data.tags.get("stoa.git_commit"),
        "metrics": {k: m[k] for k in SUMMARY_METRICS if k in m},
        "mitigations": mitigations,
        "tuning": best,
        "written": written,
        "missing": missing,
    }


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "models" / "figures"))
    ap.add_argument("--variant", choices=["rich", "portable", "both"], default="both")
    args = ap.parse_args()

    cfg = tracking.load_models_config()
    fcfg = flow_data.flow_config(cfg)
    uri = tracking.tracking_uri(cfg)
    tracking.check_server(uri)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name(tracking.experiment_name(cfg))
    if exp is None:
        raise SystemExit(f"no MLflow experiment {tracking.experiment_name(cfg)!r}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    variants = fcfg["variants"] if args.variant == "both" else [args.variant]
    summary = [export_variant(client, exp.experiment_id, v, out) for v in variants]
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    for s in summary:
        print(f"{s['variant']}: wrote {', '.join(s.get('written', [])) or 'nothing'}")
        if s.get("missing"):
            print(f"  not in MLflow: {', '.join(s['missing'])}")
        if s.get("error"):
            print(f"  {s['error']}")


if __name__ == "__main__":
    main()
