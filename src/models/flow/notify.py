"""Email the team what a stoa_train run did (rubric: notify when training finishes).

Two emails, both through src/pipeline/alerts.py (STOA_ALERT_EMAILS, smtp_default):
- gate failed: sent by src.models.flow.registry when a required variant misses a gate, with
  the reasons, so the team sees why nothing was registered (the task-failure email that
  follows only says that check_gates failed);
- training finished: sent by stoa_train's last task, `python -m src.models.flow.notify`, from
  <output_dir>/registry.json and each variant's selected.json: which versions were
  registered, and whether each became `champion` or was held back as `challenger` because it
  was worse than the champion it would have replaced.

Sending never fails the run: an email is a report, not a result.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any

from src.ingest.common import setup_logging
from src.models import tracking
from src.models.flow import data as flow_data
from src.pipeline import alerts


def _run_line() -> str:
    run_id = os.environ.get("AIRFLOW_CTX_DAG_RUN_ID")
    return f"<p>Run: {html.escape(run_id)}</p>" if run_id else ""


def _fmt(x: Any, spec: str = ".3f") -> str:
    return "n/a" if x is None else format(x, spec)


def gate_failed_email(manifest: dict[str, Any]) -> tuple[str, str]:
    """(subject, html) for a run whose gate blocked registration."""
    rows = []
    for g in manifest.get("gates", []):
        if g["passed"]:
            why = "passed"
        else:
            why = "failed: " + "; ".join(g["reasons"])
        rows.append(
            f"<li><b>{g['variant']}</b> (log_mae_cv {_fmt(g.get('log_mae_cv'))}, "
            f"{_fmt(g.get('improvement'), '.1%')} below Space Syntax, coverage80 "
            f"{_fmt(g.get('coverage80'))}): {html.escape(why)}</li>"
        )
    body = (
        "<p>stoa_train finished training, but the selected model missed a deployment gate, "
        "so <b>nothing was registered</b> and the current champion is still served.</p>"
        f"<ul>{''.join(rows)}</ul>"
        + _run_line()
        + "<p>Gates: configs/models.yaml (flow.gates). Details: docs/models/flow_model.md.</p>"
    )
    return "[Stoa] stoa_train: gate failed, nothing registered", body


def success_email(
    manifest: dict[str, Any], selected: dict[str, dict[str, Any]]
) -> tuple[str, str]:
    """(subject, html) for a run that registered at least one model. `selected` is each
    variant's selected.json."""
    rows = []
    held_back = False
    for r in manifest.get("registered", []):
        variant = r["name"].removeprefix("stoa-flow-")
        sel = selected.get(variant, {})
        m = sel.get("metrics", {})
        if r.get("kept_champion"):
            held_back = True
            outcome = (
                f"registered as v{r['version']} but <b>not promoted</b>: "
                f"{html.escape(r.get('champion_check', ''))}; champion stays "
                f"v{r.get('champion_before')}"
            )
        else:
            outcome = f"v{r['version']} is now <b>{r['alias']}</b>"
            if r.get("champion_before"):
                outcome += f" (replaces v{r['champion_before']}, now `previous`)"
        rows.append(
            f"<li><b>{r['name']}</b> ({sel.get('family', '?')}): {outcome}. "
            f"log_mae_cv {_fmt(m.get('log_mae_cv'))}, coverage80_cv "
            f"{_fmt(m.get('coverage80_cv'))}, wape_holdout {_fmt(m.get('wape_holdout'))}"
            "</li>"
        )
    if not rows:
        rows.append("<li>no variant passed its gate, so nothing was registered</li>")
    subject = "[Stoa] stoa_train finished: " + (
        "new model held back, champion kept" if held_back else "model registered"
    )
    body = (
        "<p>stoa_train finished: trained, selected, bias-checked, gated and registered.</p>"
        f"<ul>{''.join(rows)}</ul>"
        + _run_line()
        + "<p>Roll back a promotion with <code>python -m src.models.flow.registry "
        "--rollback --variant &lt;variant&gt;</code>. Models: MLflow, experiment "
        "stoa-flow.</p>"
    )
    return subject, body


def send(subject: str, body: str, what: str = "the training result") -> bool:
    return alerts.send_email(subject, body, what=what)


def load_selected(out: Path, variants: list[str]) -> dict[str, dict[str, Any]]:
    found = {}
    for v in variants:
        path = out / v / "selected.json"
        if path.exists():
            found[v] = json.loads(path.read_text())
    return found


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output-dir")
    args = ap.parse_args()
    fcfg = flow_data.flow_config(tracking.load_models_config())
    out = flow_data.output_dir(fcfg, args.output_dir)
    manifest = json.loads((out / "registry.json").read_text())
    send(*success_email(manifest, load_selected(out, fcfg["variants"])))


if __name__ == "__main__":
    main()
