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

`--promote` also compares the new model with the current champion: its gain over Space Syntax
(1 - log_mae_cv / Space Syntax's, recorded on each version as `stoa.improvement`) may be at
most `gates.champion_tolerance` below the champion's. Gains, not raw errors, because runs on
newer data or another fold plan are scored on different rows. A worse model is registered as
`challenger`, the champion stays, and the command exits 1, so the training DAG fails and
alerts.

The gate (configs/models.yaml, flow.gates) compares the latest selected run with the latest
baseline_space_syntax run on the same data version (stoa.dvc.processed) and fold plan
(stoa.plan): log_mae_cv must be at least 15% lower, and coverage80_cv within [0.75, 0.85].
If a variant in `required_variants` fails, nothing is registered and the command exits 1;
other variants that fail are reported and left out.

Writes <output_dir>/registry.json: what was registered or promoted. Building and pushing the
serving image is the deploy step's job, not this one.
"""

from __future__ import annotations

import argparse
import json
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

CHAMPION, PREVIOUS, CHALLENGER = "champion", "previous", "challenger"
IMPROVEMENT_TAG, PREVIOUS_TAG = "stoa.improvement", "stoa.previous_champion"


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


def gate(client: MlflowClient, experiment_id: str, variant: str, gates: dict) -> Gate:
    sel = latest_run(
        client,
        experiment_id,
        ["tags.`stoa.role` = 'selected'", f"tags.`stoa.variant` = '{variant}'"],
    )
    if sel is None:
        return Gate(variant, False, ["no selected run; run src.models.flow.select"])
    tags = sel.data.tags
    base = latest_run(
        client,
        experiment_id,
        [
            "tags.`stoa.family` = 'baseline_space_syntax'",
            f"tags.`stoa.dvc.processed` = '{tags.get('stoa.dvc.processed', '')}'",
            f"tags.`stoa.plan` = '{tags.get('stoa.plan', '')}'",
        ],
    )
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
    )


def alias_version(client: MlflowClient, name: str, alias: str) -> str | None:
    try:
        return client.get_model_version_by_alias(name, alias).version
    except MlflowException:
        return None


def champion_gain(client: MlflowClient, name: str) -> tuple[str | None, float | None]:
    """The champion's version and its recorded gain over Space Syntax (None if unknown)."""
    version = alias_version(client, name, CHAMPION)
    if version is None:
        return None, None
    tag = client.get_model_version(name, version).tags.get(IMPROVEMENT_TAG)
    return version, float(tag) if tag not in (None, "") else None


def beats_champion(
    improvement: float | None, champion: float | None, tolerance: float
) -> bool:
    """True unless both gains are known and the new one is more than `tolerance` lower."""
    if improvement is None or champion is None:
        return True
    return improvement >= champion - tolerance


def register(
    client: MlflowClient, g: Gate, promote: bool, tolerance: float = 0.0
) -> dict[str, Any]:
    """Register the gated run's model; with promote, make it champion if it is no worse than
    the current champion (else it becomes challenger and `held_back` is set)."""
    run = client.get_run(g.run_id)
    tags = run.data.tags
    uri = (
        tags.get("stoa.bundle_uri")
        or tags.get("stoa.model_uri")
        or f"runs:/{g.run_id}/model"
    )
    name = model_name(g.variant)
    version = mlflow.register_model(uri, name).version
    if g.improvement is not None:
        client.set_model_version_tag(name, version, IMPROVEMENT_TAG, str(g.improvement))
    out: dict[str, Any] = {"name": name, "version": version, "run_id": g.run_id}
    if promote:
        old, old_gain = champion_gain(client, name)
        out |= {"improvement": g.improvement, "champion_improvement": old_gain}
        if old is not None and not beats_champion(g.improvement, old_gain, tolerance):
            client.set_registered_model_alias(name, CHALLENGER, version)
            return out | {"alias": CHALLENGER, "held_back": True, "champion": old}
        if old is not None and old != version:
            client.set_registered_model_alias(name, PREVIOUS, old)
            client.set_model_version_tag(name, version, PREVIOUS_TAG, old)
        client.set_registered_model_alias(name, CHAMPION, version)
        return out | {"alias": CHAMPION, "held_back": False}
    client.set_registered_model_alias(name, CHALLENGER, version)
    return out | {"alias": CHALLENGER}


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
            sys.exit(1)
        if args.check:
            return
        tolerance = float(fcfg["gates"].get("champion_tolerance", 0.0))
        manifest["registered"] = [
            register(client, g, args.promote, tolerance) for g in gates if g.passed
        ]
        held = [r for r in manifest["registered"] if r.get("held_back")]
        for r in held:
            print(
                f"{r['name']}: not promoted, its gain over Space Syntax "
                f"({r['improvement']:.1%}) is below the champion's "
                f"({r['champion_improvement']:.1%}); registered as {CHALLENGER}"
            )
        if held:
            write_manifest(out, manifest)
            sys.exit(1)
    write_manifest(out, manifest)


def write_manifest(out, manifest: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "registry.json").write_text(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
