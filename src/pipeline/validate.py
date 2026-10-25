"""Validate the hourly counts table before it becomes a DVC version.

Usage: python -m src.pipeline.validate [--schema-diff]

Runs as the `validate` task of stoa_pipeline, between join_all and dvc_push. Settings and
thresholds are in configs/validation.yaml; docs/pipeline/validation.md lists every check.

Each run:
1. checks the table against the schema in configs/schemas/counts_hourly.yaml (columns and
   types);
2. runs a Great Expectations suite: keys present and unique, values in range, allowed values,
   columns missing only where the data docs allow it;
3. looks for anomalies: per-sensor outliers, stuck sensors, outage share, missing recent hours
   per sensor, stale data;
4. profiles every column and compares the profile with the previous good run (drift);
5. writes the profile and a report of every check to data/processed/validation/<site>/, and
   appends the profile summary to a history file, so DVC versions the statistics with the data.

Checks marked `error` fail the task: Airflow sends the failure email and dvc_push does not run,
so a bad table never becomes a version. `warning` checks are logged, reported and emailed
to STOA_ALERT_EMAILS (with the smtp_default settings, read without importing Airflow); the
run carries on. Outliers and stuck sensors only warn about hours
added since the last good run, so a known event doesn't alert every day.

The expected schema is generated from src/features/schema.py (`python -m src.features.schema
--write`). After an intended change to the table's columns, declare them there; `--schema-diff`
lists how the current table differs from the committed schema, to see what to declare.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import smtplib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.ingest.common import REPO_ROOT, setup_logging

# Great Expectations sends usage statistics unless told not to; set before importing it.
os.environ.setdefault("GX_ANALYTICS_ENABLED", "false")

import great_expectations as gx
import great_expectations.expectations as gxe
from great_expectations.data_context.types.base import ProgressBarsConfig
from great_expectations.expectations.row_conditions import Column, Condition

log = logging.getLogger("stoa.pipeline")

ERROR = "error"
WARNING = "warning"


class ValidationError(RuntimeError):
    """The table failed at least one error-level check."""


@dataclass
class Check:
    name: str
    severity: str  # ERROR or WARNING
    passed: bool
    detail: str = ""
    observed: dict[str, Any] = field(default_factory=dict)


def load_validation_config(path: Path | None = None) -> dict[str, Any]:
    """The validation settings, with {site} in the table and reports paths filled in."""
    with open(path or REPO_ROOT / "configs" / "validation.yaml") as f:
        cfg = yaml.safe_load(f)
    for key in ("table", "reports_dir"):
        cfg[key] = cfg[key].format(site=cfg["site"])
    return cfg


# --- Schema ---


def infer_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Column names and pandas dtypes, in table order."""
    return {"columns": {c: str(df[c].dtype) for c in df.columns}}


def schema_diff(df: pd.DataFrame, schema: dict[str, Any]) -> list[str]:
    """How the table differs from the schema: columns missing, unexpected or of another type."""
    want, have = schema["columns"], infer_schema(df)["columns"]
    out = [f"missing: {c} ({t})" for c, t in want.items() if c not in have]
    out += [f"unexpected: {c} ({t})" for c, t in have.items() if c not in want]
    out += [
        f"type: {c} is {have[c]}, schema says {t}"
        for c, t in want.items()
        if c in have and have[c] != t
    ]
    return out


def check_schema(df: pd.DataFrame, schema: dict[str, Any]) -> list[Check]:
    expected: dict[str, str] = schema["columns"]
    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]
    changed = {
        c: {"expected": t, "found": str(df[c].dtype)}
        for c, t in expected.items()
        if c in df.columns and str(df[c].dtype) != t
    }
    return [
        Check(
            "schema: columns",
            ERROR,
            not missing and not extra,
            f"missing {missing}, unexpected {extra}" if missing or extra else "",
            {"missing": missing, "unexpected": extra},
        ),
        Check(
            "schema: types",
            ERROR,
            not changed,
            ", ".join(
                f"{c} is {v['found']}, expected {v['expected']}"
                for c, v in changed.items()
            ),
            changed,
        ),
    ]


# --- Great Expectations suite ---

# Row conditions are Condition objects (Column("x") == 1): GX 2.0 removes text conditions.
OPERATORS = {
    "==": lambda c, v: c == v,
    "!=": lambda c, v: c != v,
    ">": lambda c, v: c > v,
    ">=": lambda c, v: c >= v,
    "<": lambda c, v: c < v,
    "<=": lambda c, v: c <= v,
}


def condition(rule: list[Any]) -> Condition:
    """[column, operator, value] from the config as a GX row condition."""
    column, op, value = rule
    if op not in OPERATORS:
        raise ValueError(
            f"unknown operator {op!r} in {rule}; use one of {list(OPERATORS)}"
        )
    return OPERATORS[op](Column(column), value)


def build_suite(cfg: dict[str, Any], columns: list[str]) -> list[gxe.Expectation]:
    """Expectations from the config; meta carries each one's name and severity."""

    def meta(name: str, severity: str = ERROR) -> dict[str, str]:
        return {"name": name, "severity": severity}

    exps: list[gxe.Expectation] = [
        gxe.ExpectCompoundColumnsToBeUnique(
            column_list=cfg["keys"], meta=meta("keys: unique " + "+".join(cfg["keys"]))
        )
    ]
    for col in cfg["never_missing"]:
        exps.append(
            gxe.ExpectColumnValuesToNotBeNull(column=col, meta=meta(f"present: {col}"))
        )
    for col, rule in cfg["required_when"].items():
        where = " ".join(str(part) for part in rule)
        exps.append(
            gxe.ExpectColumnValuesToNotBeNull(
                column=col,
                row_condition=condition(rule),
                meta=meta(f"present: {col} where {where}"),
            )
        )
    for col, (lo, hi) in cfg["ranges"].items():
        exps.append(
            gxe.ExpectColumnValuesToBeBetween(
                column=col,
                min_value=lo,
                max_value=hi,
                meta=meta(f"range: {col} in [{lo}, {hi}]"),
            )
        )
    for col, values in cfg["allowed_values"].items():
        exps.append(
            gxe.ExpectColumnValuesToBeInSet(
                column=col, value_set=values, meta=meta(f"allowed: {col} in {values}")
            )
        )
    share = cfg["weather_min_present"]
    for col in (c for c in columns if c.startswith("wx_")):
        exps.append(
            gxe.ExpectColumnValuesToNotBeNull(
                column=col,
                mostly=share,
                meta=meta(f"present: {col} (>= {share:.0%})", WARNING),
            )
        )
    # Skip expectations on columns the table lacks; the schema check already reports those.
    return [e for e in exps if getattr(e, "column", None) in (None, *columns)]


def run_expectations(df: pd.DataFrame, cfg: dict[str, Any]) -> list[Check]:
    context = gx.get_context(mode="ephemeral")
    context.variables.progress_bars = ProgressBarsConfig(globally=False)
    batch = (
        context.data_sources.add_pandas("stoa")
        .add_dataframe_asset("counts_hourly")
        .add_batch_definition_whole_dataframe("all_rows")
        .get_batch(batch_parameters={"dataframe": df})
    )
    suite = context.suites.add(gx.ExpectationSuite(name="counts_hourly"))
    for exp in build_suite(cfg, list(df.columns)):
        suite.add_expectation(exp)
    # BOOLEAN_ONLY is about twice as fast on 6M rows; only failures need counts and examples,
    # so those are re-run with SUMMARY.
    result = batch.validate(suite, result_format="BOOLEAN_ONLY")
    results = [
        r
        if r.success or r.exception_info.get("raised_exception")
        else batch.validate(
            r.expectation_config.to_domain_obj(), result_format="SUMMARY"
        )
        for r in result.results
    ]

    checks = []
    for r in results:
        meta = r.expectation_config.meta
        res = r.result or {}
        failed = res.get("unexpected_count")
        raised = r.exception_info.get("raised_exception")
        detail = ""
        if raised:
            detail = f"could not run: {r.exception_info.get('exception_message')}"
        elif not r.success:
            examples = res.get("partial_unexpected_list", [])[:5]
            detail = f"{failed} rows ({res.get('unexpected_percent', 0):.3f}%), e.g. {examples}"
        checks.append(
            Check(
                meta["name"],
                meta["severity"],
                bool(r.success) and not raised,
                detail,
                {"unexpected_count": failed} if failed is not None else {},
            )
        )
    return checks


# --- Anomalies ---


def outliers(df: pd.DataFrame, factor: float, since: pd.Timestamp) -> Check:
    """New hours (after `since`) above factor x their sensor's 99.9th percentile.

    The percentile comes from the sensor's whole history. Older hits are known (Run Melbourne,
    New Year's Eve) and only counted, so they don't warn, and alert, on every run.
    """
    p999 = df.groupby("location_id")["pedestriancount"].transform(
        lambda s: s.quantile(0.999)
    )
    hits = df[(p999 > 0) & (df["pedestriancount"] > factor * p999)]
    hit = hits[hits["ts_utc"] > since]
    by_sensor = hit.groupby("location_id").size().sort_values(ascending=False).head(10)
    return Check(
        f"anomaly: new counts above {factor}x the sensor's 99.9th percentile",
        WARNING,
        hit.empty,
        f"{len(hit)} rows since {since}; most at sensors {by_sensor.to_dict()}"
        if len(hit)
        else "",
        {
            "rows": len(hit),
            "by_sensor": {int(k): int(v) for k, v in by_sensor.items()},
            "earlier_rows": len(hits) - len(hit),
            "since": str(since),
        },
    )


def stuck_sensors(
    df: pd.DataFrame, run_hours: int, min_count: int, since: pd.Timestamp
) -> Check:
    """Runs of the same count (at least min_count) in consecutive hours at one sensor, that
    reach into the new hours (after `since`); older runs are only counted.

    Small counts repeat naturally overnight (1 an hour for six hours), so they are ignored.
    """
    d = df[["location_id", "ts_utc", "pedestriancount"]].sort_values(
        ["location_id", "ts_utc"]
    )
    new_run = (
        d["location_id"].ne(d["location_id"].shift())
        | d["ts_utc"].diff().ne(pd.Timedelta(hours=1))
        | d["pedestriancount"].ne(d["pedestriancount"].shift())
    )
    d["run"] = new_run.cumsum()
    runs = d.groupby("run").agg(
        location_id=("location_id", "first"),
        count=("pedestriancount", "first"),
        hours=("pedestriancount", "size"),
        start=("ts_utc", "first"),
        end=("ts_utc", "last"),
    )
    long_runs = runs[(runs["count"] >= min_count) & (runs["hours"] >= run_hours)]
    stuck = long_runs[long_runs["end"] > since]
    sensors = sorted(int(s) for s in stuck["location_id"].unique())
    return Check(
        f"anomaly: new runs of the same count (>= {min_count}) for {run_hours}+ hours",
        WARNING,
        stuck.empty,
        f"{len(stuck)} runs at sensors {sensors}, longest {stuck['hours'].max()} hours"
        if len(stuck)
        else "",
        {
            "runs": len(stuck),
            "sensors": sensors,
            "earlier_runs": len(long_runs) - len(stuck),
        },
    )


def outage_share(df: pd.DataFrame, max_share: float) -> Check:
    share = float(df["suspect_outage_zero"].mean())
    return Check(
        f"anomaly: suspected outage hours <= {max_share:.1%}",
        WARNING,
        share <= max_share,
        f"{share:.2%} of rows" if share > max_share else "",
        {"share": round(share, 5)},
    )


def recent_completeness(
    df: pd.DataFrame, days: int, min_share: float, daytime: tuple[int, int]
) -> Check:
    """Share of the last `days` of daytime live hours each sensor reported, and silent sensors.

    The live portal leaves out hours with no one counted, so night hours go missing at quiet
    sensors; only daytime hours (when a working sensor always counts someone) are expected.
    """
    live = df.loc[df["source"] == "live", ["location_id", "ts_utc", "hour_local"]]
    if live.empty:
        return Check("anomaly: recent hours per sensor", WARNING, True, "no live rows")
    end = live["ts_utc"].max()
    start = end - pd.Timedelta(days=days)
    recent = live[live["ts_utc"] > start]
    recent = recent[recent["hour_local"].between(daytime[0], daytime[1] - 1)]
    before = live[
        (live["ts_utc"] <= start) & (live["ts_utc"] > start - pd.Timedelta(days=30))
    ]
    # Expected hours: every daytime hour in the window that any sensor reported.
    expected = recent["ts_utc"].nunique()
    share = recent.groupby("location_id").size() / max(expected, 1)
    low = {int(k): round(float(v), 2) for k, v in share[share < min_share].items()}
    silent = sorted(
        int(s) for s in set(before["location_id"]) - set(recent["location_id"])
    )
    problems = []
    if low:
        problems.append(f"under {min_share:.0%} of daytime hours: {low}")
    if silent:
        problems.append(f"reported in the 30 days before but not since: {silent}")
    return Check(
        f"anomaly: each sensor reported >= {min_share:.0%} of the last {days} days' "
        "daytime hours",
        WARNING,
        not problems,
        "; ".join(problems),
        {
            "low": low,
            "silent": silent,
            "window_end": str(end),
            "expected_hours": expected,
        },
    )


def freshness(df: pd.DataFrame, max_age_days: float, now: datetime) -> Check:
    newest = df["ts_utc"].max()
    age_days = (pd.Timestamp(now) - newest) / pd.Timedelta(days=1)
    return Check(
        f"anomaly: newest hour at most {max_age_days} days old",
        WARNING,
        age_days <= max_age_days,
        f"newest hour is {newest} ({age_days:.1f} days old)"
        if age_days > max_age_days
        else "",
        {"newest": str(newest), "age_days": round(float(age_days), 2)},
    )


def new_since(
    df: pd.DataFrame, previous: dict[str, Any] | None, days: int
) -> pd.Timestamp:
    """Where the new hours start: after the previous good run's newest hour, or, with no
    previous run, the last `days` days."""
    if previous is not None:
        return pd.Timestamp(previous["last_hour_utc"])
    return df["ts_utc"].max() - pd.Timedelta(days=days)


def anomaly_checks(
    df: pd.DataFrame, cfg: dict[str, Any], now: datetime, since: pd.Timestamp
) -> list[Check]:
    a = cfg["anomalies"]
    return [
        outliers(df, a["outlier_factor"], since),
        stuck_sensors(df, a["stuck_run_hours"], a["stuck_min_count"], since),
        outage_share(df, a["max_outage_share"]),
        recent_completeness(
            df,
            a["recent_days"],
            a["min_recent_completeness"],
            tuple(a["daytime_hours"]),
        ),
        freshness(df, a["max_age_days"], now),
    ]


# --- Statistics and drift ---


def _number(x: Any) -> float | None:
    return None if pd.isna(x) else round(float(x), 4)


def profile(df: pd.DataFrame) -> dict[str, Any]:
    """Per-column statistics plus a table summary; saved every run to compare over time."""
    columns: dict[str, dict[str, Any]] = {}
    for c in df.columns:
        s = df[c]
        stats: dict[str, Any] = {
            "dtype": str(s.dtype),
            "null_share": _number(s.isna().mean()),
        }
        if pd.api.types.is_bool_dtype(s):
            stats["true_share"] = _number(s.mean())
        elif pd.api.types.is_numeric_dtype(s):
            stats |= {
                k: _number(v)
                for k, v in s.agg(["min", "max", "mean", "std", "median"]).items()
            }
        elif pd.api.types.is_datetime64_any_dtype(s):
            stats |= {"min": str(s.min()), "max": str(s.max())}
        else:
            stats["distinct"] = int(s.nunique())
            stats["top"] = {str(k): int(v) for k, v in s.value_counts().head(5).items()}
        columns[c] = stats
    return {
        "rows": len(df),
        "sensors": int(df["location_id"].nunique()),
        "first_hour_utc": str(df["ts_utc"].min()),
        "last_hour_utc": str(df["ts_utc"].max()),
        "mean_count": _number(df["pedestriancount"].mean()),
        "columns": columns,
    }


def summary(stats: dict[str, Any], run_at: datetime) -> dict[str, Any]:
    """One line of the history file: the table summary and each column's missing share."""
    return {
        "run_at": run_at.isoformat(timespec="seconds"),
        **{k: stats[k] for k in ("rows", "sensors", "first_hour_utc", "last_hour_utc")},
        "mean_count": stats["mean_count"],
        "null_share": {c: v["null_share"] for c, v in stats["columns"].items()},
    }


def drift_checks(
    current: dict[str, Any], previous: dict[str, Any] | None, cfg: dict[str, Any]
) -> list[Check]:
    if previous is None:
        return [
            Check(
                "drift: compared with the previous run",
                WARNING,
                True,
                "no previous run",
            )
        ]
    d = cfg["drift"]
    checks = []

    drop = (previous["rows"] - current["rows"]) / previous["rows"]
    checks.append(
        Check(
            f"drift: rows dropped by at most {d['max_row_drop_share']:.0%}",
            ERROR,
            drop <= d["max_row_drop_share"],
            f"{previous['rows']} -> {current['rows']} rows ({drop:.1%} fewer)"
            if drop > d["max_row_drop_share"]
            else "",
            {"previous": previous["rows"], "current": current["rows"]},
        )
    )
    checks.append(
        Check(
            "drift: no fewer sensors",
            WARNING,
            current["sensors"] >= previous["sensors"],
            f"{previous['sensors']} -> {current['sensors']} sensors"
            if current["sensors"] < previous["sensors"]
            else "",
            {"previous": previous["sensors"], "current": current["sensors"]},
        )
    )

    limit = d["max_null_increase_pp"]
    grew = {}
    for c, now_share in current["null_share"].items():
        before = previous["null_share"].get(c)
        if (
            now_share is not None
            and before is not None
            and (now_share - before) * 100 > limit
        ):
            grew[c] = {"previous": before, "current": now_share}
    checks.append(
        Check(
            f"drift: missing values up by at most {limit} points per column",
            WARNING,
            not grew,
            ", ".join(
                f"{c} {v['previous']:.1%} -> {v['current']:.1%}"
                for c, v in grew.items()
            ),
            grew,
        )
    )

    change = (
        abs(current["mean_count"] - previous["mean_count"]) / previous["mean_count"]
    )
    checks.append(
        Check(
            f"drift: mean hourly count changed by at most {d['max_mean_count_change']:.0%}",
            WARNING,
            change <= d["max_mean_count_change"],
            f"{previous['mean_count']} -> {current['mean_count']}"
            if change > d["max_mean_count_change"]
            else "",
            {"previous": previous["mean_count"], "current": current["mean_count"]},
        )
    )
    return checks


def last_summary(history: Path) -> dict[str, Any] | None:
    if not history.is_file():
        return None
    lines = [line for line in history.read_text().splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else None


# --- Run ---


def validate(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    schema: dict[str, Any],
    previous: dict[str, Any] | None,
    now: datetime,
) -> tuple[list[Check], dict[str, Any]]:
    """Every check on df, and its statistics profile."""
    checks = check_schema(df, schema)
    checks += run_expectations(df, cfg)
    since = new_since(df, previous, cfg["anomalies"]["recent_days"])
    checks += anomaly_checks(df, cfg, now, since)
    stats = profile(df)
    checks += drift_checks(summary(stats, now), previous, cfg)
    return checks, stats


def write_reports(
    out: Path, checks: list[Check], stats: dict[str, Any], now: datetime
) -> None:
    """The profile and the check report (overwritten each run); history gains a line on success."""
    out.mkdir(parents=True, exist_ok=True)
    errors = [c for c in checks if not c.passed and c.severity == ERROR]
    warnings = [c for c in checks if not c.passed and c.severity == WARNING]
    report = {
        "run_at": now.isoformat(timespec="seconds"),
        "passed": not errors,
        "errors": len(errors),
        "warnings": len(warnings),
        "checks": [asdict(c) for c in checks],
    }
    (out / "counts_hourly_validation.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    (out / "counts_hourly_stats.json").write_text(
        json.dumps(stats, indent=2, default=str) + "\n"
    )
    # Only good runs become the baseline the next run is compared with.
    if not errors:
        with open(out / "counts_hourly_stats_history.jsonl", "a") as f:
            f.write(json.dumps(summary(stats, now), default=str) + "\n")


# --- Warning alert ---


def warning_email(
    checks: list[Check], run_id: str | None = None, report: str | None = None
) -> tuple[str, str] | None:
    """Subject and HTML body listing the failed warning checks, or None if there are none."""
    warnings = [c for c in checks if not c.passed and c.severity == WARNING]
    if not warnings:
        return None
    subject = (
        f"[Stoa] validate: {len(warnings)} data warning{'s' * (len(warnings) > 1)}"
    )
    rows = "".join(f"<li><b>{c.name}</b>: {c.detail}</li>" for c in warnings)
    body = (
        "<p>The hourly table passed every error check, so the run carried on and the data "
        "is being versioned, but these warnings need a look:</p>"
        f"<ul>{rows}</ul>"
        + (f"<p>Run: {run_id}</p>" if run_id else "")
        + (f"<p>Full report: <code>{report}</code>. " if report else "<p>")
        + "Checks: docs/pipeline/validation.md.</p>"
    )
    return subject, body


def smtp_settings() -> dict[str, Any] | None:
    """The smtp_default connection that docker-compose.yaml passes in AIRFLOW_CONN_SMTP_DEFAULT
    (JSON), the one the failure emails use, or None if it isn't set or has no login.

    Read directly rather than through Airflow: inside a task, Airflow 3 hides the metadata
    database from the task's process, so importing airflow there fails.
    """
    raw = os.environ.get("AIRFLOW_CONN_SMTP_DEFAULT", "")
    try:
        conn = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        log.warning(
            "AIRFLOW_CONN_SMTP_DEFAULT is not JSON, so the warnings were not emailed"
        )
        return None
    if not conn.get("host") or not conn.get("login"):
        return None
    return conn


def send_warning_alert(checks: list[Check], report: str | None = None) -> bool:
    """Email the warnings to STOA_ALERT_EMAILS through the smtp_default settings. Returns
    whether an email went out.

    Nothing here raises: warnings never stop the pipeline, so a failed send is only logged.
    """
    try:
        message = warning_email(
            checks, os.environ.get("AIRFLOW_CTX_DAG_RUN_ID"), report
        )
        recipients = [
            e.strip()
            for e in os.environ.get("STOA_ALERT_EMAILS", "").split(",")
            if e.strip()
        ]
        if message is None:
            return False
        if not recipients:
            log.info("STOA_ALERT_EMAILS is empty, so the warnings were not emailed")
            return False
        conn = smtp_settings()
        if conn is None:
            log.info(
                "no SMTP settings (AIRFLOW_CONN_SMTP_DEFAULT), so the warnings were not emailed"
            )
            return False
        subject, body = message
        sender = os.environ.get("STOA_SMTP_USER") or conn["login"]
        mail = EmailMessage()
        mail["Subject"] = subject
        mail["From"] = sender
        mail["To"] = ", ".join(recipients)
        mail.set_content("This email needs an HTML-capable mail client.")
        mail.add_alternative(body, subtype="html")
        with smtplib.SMTP(
            conn["host"], int(conn.get("port") or 587), timeout=30
        ) as smtp:
            # Gmail on 587: plain connection upgraded with STARTTLS (disable_ssl in extra).
            smtp.starttls()
            smtp.login(conn["login"], conn.get("password") or "")
            smtp.send_message(mail)
    except Exception:
        log.exception("could not email the warnings")
        return False
    log.info("emailed the warnings to %s", ", ".join(recipients))
    return True


def log_checks(checks: list[Check]) -> None:
    for c in checks:
        if c.passed:
            log.info("PASS  %s", c.name)
        elif c.severity == ERROR:
            log.error("FAIL  %s: %s", c.name, c.detail)
        else:
            log.warning("WARN  %s: %s", c.name, c.detail)
    errors = sum(not c.passed and c.severity == ERROR for c in checks)
    warnings = sum(not c.passed and c.severity == WARNING for c in checks)
    log.info("%d checks: %d errors, %d warnings", len(checks), errors, warnings)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--schema-diff",
        action="store_true",
        help="list how the current table differs from the committed schema",
    )
    args = parser.parse_args()

    cfg = load_validation_config()
    table = REPO_ROOT / cfg["table"]
    if not table.is_file():
        raise ValidationError(f"{table} does not exist; run join_all first")
    df = pd.read_parquet(table)
    log.info("read %s: %d rows, %d columns", cfg["table"], len(df), df.shape[1])
    schema_path = REPO_ROOT / cfg["schema"]
    schema = yaml.safe_load(schema_path.read_text())
    if args.schema_diff:
        diff = schema_diff(df, schema)
        print("\n".join(diff) if diff else "the table matches the schema")
        return

    out = REPO_ROOT / cfg["reports_dir"]
    now = datetime.now(UTC)
    checks, stats = validate(
        df, cfg, schema, last_summary(out / "counts_hourly_stats_history.jsonl"), now
    )
    write_reports(out, checks, stats, now)
    log_checks(checks)
    log.info(
        "report: %s", (out / "counts_hourly_validation.json").relative_to(REPO_ROOT)
    )

    failed = [c.name for c in checks if not c.passed and c.severity == ERROR]
    if failed:
        # The failure email covers this run; no separate warning email.
        raise ValidationError(f"{len(failed)} checks failed: {'; '.join(failed)}")
    send_warning_alert(checks, f"{cfg['reports_dir']}/counts_hourly_validation.json")


if __name__ == "__main__":
    main()
