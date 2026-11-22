"""Gate the selected flow models, register them, and move the serving alias.

Usage:
  python -m src.models.flow.registry --check              # gates only; exit 1 if one fails
  python -m src.models.flow.registry                      # gate, register as `challenger`
  python -m src.models.flow.registry --promote            # gate, register as `champion`
  python -m src.models.flow.registry --rollback --variant rich

Each variant is its own registered model, `stoa-flow-<variant>`. Aliases, not stages (MLflow 3
deprecates stages): `champion` is the version to serve, `previous` the one it replaced, and
rollback moves `champion` back to `previous`. Each champion version records the one it
replaced (tag `stoa.previous_champion`), so rollback can be repeated, one version at a time.
A rollback of several variants changes none of them unless every one has a `previous`.

What is registered is the selected run's interval bundle (`stoa.bundle_uri`, see
src/models/flow/bundle.py: predicted, lower and upper per row), or its mean model for runs
from before the bundle existed (`stoa.model_uri`).

The gate (configs/models.yaml, flow.gates) compares the latest selected run with the latest
baseline_space_syntax run on the same data version (stoa.dvc.processed) and fold plan
(stoa.plan): log_mae_cv must be at least 10% lower (gates.log_mae_improvement), and
coverage80_cv within [0.75, 0.85].
If a variant in `required_variants` fails, nothing is registered, the team is emailed the
reasons (src/models/flow/notify.py) and the command exits 1; other variants that fail are
reported and left out.

A model that passes must also beat the current champion before it replaces it. With
`--promote`, a new version whose log_mae_cv is worse than the champion's (by more than
`gates.champion_tolerance`) is registered as `challenger` and `champion` stays where it is, so
a worse retrain is never served. `--rollback` undoes a promotion by hand.

Scores are only comparable on the same test: a fold plan that holds out a buffer around each
test sensor is harder, and the same Space Syntax baseline scores worse on it. So when the
champion was scored on another fold plan or data version, each model's log_mae_cv is divided
by its own Space Syntax baseline's (the ratio the gate uses) and those ratios are compared. If
either baseline is missing, the raw scores are compared, as before.

Writes <output_dir>/registry.json: what was registered or promoted. Building and pushing the
serving image is the deploy step's job, not this one.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.ingest.common import setup_logging
from src.models import tracking
from src.models.flow import data as flow_data
from src.models.flow import notify

log = logging.getLogger("stoa.models")

CHAMPION, PREVIOUS, CHALLENGER = "champion", "previous", "challenger"
PREVIOUS_TAG = "stoa.previous_champion"


def model_name(variant: str) -> str:
    return f"stoa-flow-{variant}"


@dataclass
class Gate:
    variant: str
    passed: bool
    reasons: list[str]
    run_id: str | None = None
    baseline_run_id: str | None = None
    improvement: float | None = None
    coverage80: float | None = None
    log_mae_cv: float | None = None


def evaluate_gate(
    selected: dict[str, float],
    baseline: dict[str, float] | None,
    gates: dict[str, Any],
) -> tuple[bool, list[str], float | None]:
    """(passed, reasons it failed, improvement over Space Syntax) from the runs' metrics."""
    reasons = []
    improvement = None
    if baseline is None or not baseline.get("log_mae_cv"):
        reasons.append(
            "no baseline_space_syntax run on the same data version and fold plan"
        )
    else:
        improvement = 1 - selected["log_mae_cv"] / baseline["log_mae_cv"]
        if improvement < gates["log_mae_improvement"]:
            reasons.append(
                f"log_mae_cv is {improvement:.1%} below Space Syntax "
                f"(needs {gates['log_mae_improvement']:.0%})"
            )
    low, high = gates["coverage80"]
    cov = selected.get("coverage80_cv")
    if cov is None or not low <= cov <= high:
        reasons.append(f"coverage80_cv {cov} is outside [{low}, {high}]")
    limit = gates.get("max_worst_slice_ratio")
    ratio = selected.get("worst_slice_ratio")
    if limit is not None and ratio is not None and ratio > limit:
        reasons.append(f"worst slice is {ratio:.2f}x the overall WAPE (limit {limit})")
    return not reasons, reasons, improvement


def latest_run(
    client: MlflowClient, experiment_id: str, filters: list[str]
) -> Any | None:
    runs = client.search_runs(
        [experiment_id],
        filter_string=" and ".join(["attributes.status = 'FINISHED'", *filters]),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    return runs[0] if runs else None


def baseline_run(client: MlflowClient, experiment_id: str, tags: dict) -> Any | None:
    """The Space Syntax run scored on the same data version and fold plan as `tags`' run."""
    return latest_run(
        client,
        experiment_id,
        [
            "tags.`stoa.family` = 'baseline_space_syntax'",
            f"tags.`stoa.dvc.processed` = '{tags.get('stoa.dvc.processed', '')}'",
            f"tags.`stoa.plan` = '{tags.get('stoa.plan', '')}'",
        ],
    )


def gate(client: MlflowClient, experiment_id: str, variant: str, gates: dict) -> Gate:
    sel = latest_run(
        client,
        experiment_id,
        ["tags.`stoa.role` = 'selected'", f"tags.`stoa.variant` = '{variant}'"],
    )
    if sel is None:
        return Gate(variant, False, ["no selected run; run src.models.flow.select"])
    base = baseline_run(client, experiment_id, sel.data.tags)
    passed, reasons, improvement = evaluate_gate(
        sel.data.metrics, base.data.metrics if base else None, gates
    )
    return Gate(
        variant,
        passed,
        reasons,
        sel.info.run_id,
        base.info.run_id if base else None,
        improvement,
        sel.data.metrics.get("coverage80_cv"),
        sel.data.metrics.get("log_mae_cv"),
    )


def alias_version(client: MlflowClient, name: str, alias: str) -> str | None:
    try:
        return client.get_model_version_by_alias(name, alias).version
    except MlflowException:
        return None


def beats_champion(
    new: float | None,
    champion: float | None,
    tolerance: float = 0.0,
    metric: str = "log_mae_cv",
) -> tuple[bool, str]:
    """(may replace the champion, why) from the two models' log_mae_cv (lower is better).

    No champion, or a champion without the metric, can't block a model that passed the gate.
    """
    if champion is None:
        return True, "no champion to compare with"
    if new is None:
        return False, "the new model has no log_mae_cv"
    if new <= champion * (1 + tolerance):
        return True, f"{metric} {new:.4f} vs champion {champion:.4f}"
    return False, (
        f"{metric} {new:.4f} is worse than the champion's {champion:.4f}"
        + (f" (tolerance {tolerance:.0%})" if tolerance else "")
    )


def champion_run(client: MlflowClient, name: str) -> tuple[str | None, Any | None]:
    """(champion version, the run it was registered from)."""
    version = alias_version(client, name, CHAMPION)
    if version is None:
        return None, None
    run_id = client.get_model_version(name, version).run_id
    return version, client.get_run(run_id) if run_id else None


COMPARE_KEYS = ("stoa.plan", "stoa.dvc.processed")


def comparable_scores(
    client: MlflowClient, new_run: Any, old_run: Any | None
) -> tuple[float | None, float | None, str]:
    """(new score, champion score, what they are), on a footing where lower is better.

    Same fold plan and data version: raw log_mae_cv. Otherwise each divided by its own Space
    Syntax baseline's, so a harder test doesn't count against the model scored on it.
    """
    new = new_run.data.metrics.get("log_mae_cv")
    if old_run is None:
        return new, None, "log_mae_cv"
    old = old_run.data.metrics.get("log_mae_cv")
    new_tags, old_tags = new_run.data.tags, old_run.data.tags
    if all(new_tags.get(k) == old_tags.get(k) for k in COMPARE_KEYS):
        return new, old, "log_mae_cv"
    new_base = baseline_run(client, new_run.info.experiment_id, new_tags)
    old_base = baseline_run(client, old_run.info.experiment_id, old_tags)
    nb = new_base.data.metrics.get("log_mae_cv") if new_base else None
    ob = old_base.data.metrics.get("log_mae_cv") if old_base else None
    if new is None or old is None or not nb or not ob:
        log.warning(
            "champion scored on another fold plan or data version, and a Space Syntax "
            "baseline is missing: comparing raw log_mae_cv"
        )
        return new, old, "log_mae_cv"
    return new / nb, old / ob, "log_mae_cv / Space Syntax"


def register(
    client: MlflowClient, g: Gate, promote: bool, tolerance: float = 0.0
) -> dict[str, Any]:
    run = client.get_run(g.run_id)
    tags = run.data.tags
    uri = (
        tags.get("stoa.bundle_uri")
        or tags.get("stoa.model_uri")
        or f"runs:/{g.run_id}/model"
    )
    name = model_name(g.variant)
    new_metric = run.data.metrics.get("log_mae_cv")
    # Read the champion before registering, so the comparison is with the served version.
    old, old_run = champion_run(client, name)
    old_metric = old_run.data.metrics.get("log_mae_cv") if old_run else None
    version = mlflow.register_model(uri, name).version
    result = {"name": name, "version": version, "run_id": g.run_id}
    if not promote:
        client.set_registered_model_alias(name, CHALLENGER, version)
        return {**result, "alias": CHALLENGER}
    new_score, old_score, basis = comparable_scores(client, run, old_run)
    better, why = beats_champion(new_score, old_score, tolerance, basis)
    result |= {
        "log_mae_cv": new_metric,
        "champion_before": old,
        "champion_log_mae_cv": old_metric,
        "champion_check": why,
        "champion_check_basis": basis,
    }
    if not better:
        # Automatic rollback guard: the worse model is kept for inspection, never served.
        client.set_registered_model_alias(name, CHALLENGER, version)
        log.warning(
            "%s v%s not promoted: %s; champion stays v%s", name, version, why, old
        )
        return {**result, "alias": CHALLENGER, "kept_champion": True}
    if old is not None and old != version:
        client.set_registered_model_alias(name, PREVIOUS, old)
        client.set_model_version_tag(name, version, PREVIOUS_TAG, old)
    client.set_registered_model_alias(name, CHAMPION, version)
    return {**result, "alias": CHAMPION, "kept_champion": False}


def rollback(client: MlflowClient, variant: str) -> dict[str, Any]:
    """Point champion back at the version it replaced, and `previous` at the one before that
    (if recorded); fails if there is no `previous`."""
    name = model_name(variant)
    prev = alias_version(client, name, PREVIOUS)
    if prev is None:
        raise SystemExit(f"{name} has no `{PREVIOUS}` version to roll back to")
    current = alias_version(client, name, CHAMPION)
    client.set_registered_model_alias(name, CHAMPION, prev)
    before = client.get_model_version(name, prev).tags.get(PREVIOUS_TAG)
    if before:
        client.set_registered_model_alias(name, PREVIOUS, before)
    else:
        client.delete_registered_model_alias(name, PREVIOUS)
    return {
        "name": name,
        "champion": prev,
        "rolled_back_from": current,
        "previous": before,
    }


def rollback_all(client: MlflowClient, variants: list[str]) -> list[dict[str, Any]]:
    """Roll back every variant, or none: all must have a `previous` before any is changed."""
    missing = [
        v for v in variants if alias_version(client, model_name(v), PREVIOUS) is None
    ]
    if missing:
        names = ", ".join(model_name(v) for v in missing)
        raise SystemExit(f"nothing rolled back: {names} has no `{PREVIOUS}` version")
    return [rollback(client, v) for v in variants]


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", choices=["rich", "portable", "both"], default="both")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="gates only")
    mode.add_argument("--promote", action="store_true", help="register as champion")
    mode.add_argument("--rollback", action="store_true")
    ap.add_argument("--output-dir")
    args = ap.parse_args()

    cfg = tracking.load_models_config()
    fcfg = flow_data.flow_config(cfg)
    uri = tracking.tracking_uri(cfg)
    tracking.check_server(uri)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    variants = fcfg["variants"] if args.variant == "both" else [args.variant]
    out = flow_data.output_dir(fcfg, args.output_dir)
    manifest: dict[str, Any] = {"at": datetime.now(UTC).isoformat(timespec="seconds")}

    if args.rollback:
        manifest["rollback"] = rollback_all(client, variants)
    else:
        name = tracking.experiment_name(cfg)
        exp = client.get_experiment_by_name(name)
        if exp is None:
            raise SystemExit(f"no MLflow experiment {name!r}")
        gates = [gate(client, exp.experiment_id, v, fcfg["gates"]) for v in variants]
        for g in gates:
            status = "passed" if g.passed else "FAILED: " + "; ".join(g.reasons)
            print(f"{model_name(g.variant)}: gate {status}")
        manifest["gates"] = [asdict(g) for g in gates]
        required = set(fcfg["gates"]["required_variants"])
        failed = [g.variant for g in gates if not g.passed and g.variant in required]
        if failed:
            # A required variant failing blocks the whole deployment, so nothing registers.
            write_manifest(out, manifest)
            notify.send(*notify.gate_failed_email(manifest), what="the gate result")
            sys.exit(1)
        if args.check:
            return
        tolerance = fcfg["gates"].get("champion_tolerance", 0.0)
        manifest["registered"] = [
            register(client, g, args.promote, tolerance) for g in gates if g.passed
        ]
    write_manifest(out, manifest)


def write_manifest(out, manifest: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "registry.json").write_text(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
