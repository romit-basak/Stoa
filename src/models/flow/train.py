"""Train the flow model's candidates and the baseline ladder on the location folds.

Usage: python -m src.models.flow.train [--variant rich|portable|both] [--families ...]
       [--tune] [--n-trials N] [--folds N] [--max-trees N] [--train-rows N]
       [--output-dir DIR]

For each variant, every candidate family (src/models/flow/candidates.py) trains on each fold's
training sample and predicts that fold's sensors, so every prediction comes from a model that
never saw the sensor or its neighbours (CLAUDE.md core rule 2). The baseline ladder
(src/models/flow/baselines.py) is fitted on the same plan. Holdout precincts are in no fold;
src.models.flow.select scores them for the selected model only.

Each model is one MLflow run in experiment stoa-flow (src/models/tracking.py records the code
and data version): `<family>_<variant>` for candidates, `baseline_<rung>` for the ladder, with
the fold metrics named as in docs/models/experiment_tracking.md. The baselines use only network
features, which both variants share, so they run once.

Writes to <output_dir>/<variant>/: oof.parquet (one prediction column per model),
candidates.json (each model's run id, parameters and metrics), and reports/<model>.md, each
model's training report (src/models/flow/report.py), also logged to its run. The baselines'
reports go to <output_dir>/baselines/reports/. src.models.flow.select reads the first two next.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from src.features.common import log
from src.ingest.common import setup_logging
from src.models import tracking
from src.models.flow import baselines, candidates
from src.models.flow import data as flow_data
from src.models.flow import report as flow_report
from src.models.flow import tune as flow_tune
from src.models.flow.data import Data
from src.models.flow.evaluate import calibrated_cv_metrics, cv_metrics, scored_mask


def fold_rows(data: Data, plan: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Row positions of every fold's test rows, in fold order."""
    return np.concatenate([t for _, t in plan])


def fold_scores(
    data: Data,
    pred: np.ndarray,
    plan: list[tuple[np.ndarray, np.ndarray]],
    calibration_days: float,
) -> dict[str, float]:
    """src.models.flow.evaluate's fold scores of out-of-fold predictions, uncalibrated
    (after each sensor's calibration window) and calibrated (`_cal`)."""
    tested = fold_rows(data, plan)
    rows = data.frame.hourly.iloc[tested][["location_id", "ts_utc"]].reset_index(
        drop=True
    )
    y, p, folds = data.y[tested], pred[tested], data.fold[tested]
    keep = scored_mask(rows["location_id"].to_numpy(), rows["ts_utc"], calibration_days)
    out = cv_metrics(rows[keep], y[keep], p[keep], folds[keep])
    out.update(calibrated_cv_metrics(rows, y, p, folds, calibration_days))
    return out


def plan_tag(n_folds: int, train_rows: int, seed: int) -> str:
    """Identifies a fold plan, so the gate only compares runs scored on the same rows."""
    return f"folds={n_folds};train_rows={train_rows};seed={seed}"


def oof_candidates(
    data: Data,
    families: list[str],
    params: dict[str, dict[str, Any]],
    plan: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    diagnostics: dict[str, dict[str, list]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Out-of-fold predictions per family, building each fold's matrices once for all.

    If `diagnostics` is given, each family's fold cost curves and parameter counts are added
    to it (`curves`, `learned`) for the training report.
    """
    n = len(data.y)
    preds = {f: np.full(n, np.nan, dtype="float32") for f in families}
    seconds = dict.fromkeys(families, 0.0)
    for k, (train, test) in enumerate(plan):
        x_train, x_test = data.frame.matrix(train), data.frame.matrix(test)
        s_tr = flow_report.sample(np.arange(len(train)), seed=seed + k)
        s_te = flow_report.sample(np.arange(len(test)), seed=seed + k)
        sets = {
            "training rows": (x_train[s_tr], data.y[train][s_tr]),
            "held-out fold sensors": (x_test[s_te], data.y[test][s_te]),
        }
        for f in families:
            t = time.perf_counter()
            model = candidates.fit(
                candidates.make(f, params[f], seed), x_train, data.y[train]
            )
            preds[f][test] = candidates.predict(model, x_test)
            seconds[f] += time.perf_counter() - t
            log.info("fold %d %s: %.0f s", k, f, time.perf_counter() - t)
            if diagnostics is not None:
                d = diagnostics.setdefault(f, {"curves": [], "learned": []})
                start = float(np.mean(data.y[train]))
                d["curves"].append(flow_report.cost_curve(f, model, sets, start))
                d["learned"].append(flow_report.learned(f, model))
        del x_train, x_test, sets
    return preds, seconds


def oof_baselines(
    data: Data,
    plan: list[tuple[np.ndarray, np.ndarray]],
    fitted: dict[str, list] | None = None,
) -> dict[str, np.ndarray]:
    """Each rung's out-of-fold predictions; its fold models go into `fitted`, if given."""
    out = {}
    for r in baselines.RUNGS:
        models: list = []
        out[f"baseline_{r}"] = baselines.oof(
            r, data.frame.hourly, data.frame.osm, data.y, plan, models
        )
        if fitted is not None:
            fitted[f"baseline_{r}"] = models
    return out


def baseline_report(
    data: Data,
    name: str,
    plan: list[tuple[np.ndarray, np.ndarray]],
    models: list,
    metrics: dict[str, float],
) -> str:
    hourly, osm = data.frame.hourly, data.frame.osm
    curves = [
        flow_report.baseline_curve(
            m,
            hourly,
            osm,
            data.y,
            {
                "training rows": flow_report.sample(tr, seed=k),
                "held-out fold sensors": flow_report.sample(te, seed=k),
            },
        )
        for k, (m, (tr, te)) in enumerate(zip(models, plan))
    ]
    train = np.unique(np.concatenate([tr for tr, _ in plan]))
    facts = flow_report.data_facts(
        data,
        train,
        flow_report.fold_hidden(data),
        train_rows=len(plan[0][0]),
        rows_label="sampled, per fold",
        note="One model per location fold, as for the candidates; costs averaged over folds.",
    )
    alphas = [float(m.ridge_.alpha_) for m in models if hasattr(m.ridge_, "alpha_")]
    return flow_report.render(
        name,
        f"Baseline rung `{name.removeprefix('baseline_')}`: the hour-of-week profile of log "
        "counts plus a ridge regression of each sensor's mean residual on its network terms "
        f"({', '.join(models[0].columns[:6])}{' ...' if len(models[0].columns) > 6 else ''}).",
        facts,
        "squared error on log1p counts",
        "mean squared error on log1p counts over the rows (sampled); the ridge also adds "
        "alpha times the squared coefficients, fitted on one row per sensor",
        {
            "Cost, fitted in one step (no trees), mean over folds": flow_report.mean_curve(
                curves
            )
        },
        flow_report.mean_counts([flow_report.baseline_learned(m) for m in models]),
        {
            "alpha_grid": f"{baselines.ALPHAS.min():g} to {baselines.ALPHAS.max():g} "
            f"({len(baselines.ALPHAS)} values, log scale)",
            "alpha_chosen_per_fold": ", ".join(f"{a:g}" for a in alphas),
            "network_terms": len(models[0].columns),
        },
        "The ridge penalty alpha is chosen in each fold by leave-one-out over the training "
        "sensors.",
        metrics,
        datetime.now(UTC).isoformat(timespec="seconds"),
    )


def log_run(
    name: str,
    params: dict[str, Any],
    tags: dict[str, str],
    metrics: dict[str, float],
    report: str | None = None,
) -> str:
    with tracking.run(name, params=params, tags=tags) as active:
        mlflow.log_metrics(metrics)
        if report:
            mlflow.log_text(report, "model_report.md")
        return active.info.run_id


FAMILY_NAMES = {
    "hgb": "scikit-learn HistGradientBoostingRegressor",
    "lgbm": "LightGBM",
    "xgb": "XGBoost (histogram trees)",
    "rf": "scikit-learn RandomForestRegressor",
}


def write_report(out: Path, name: str, text: str) -> None:
    path = out / "reports" / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def candidate_report(
    data: Data,
    family: str,
    params: dict[str, Any],
    plan: list[tuple[np.ndarray, np.ndarray]],
    diag: dict[str, list],
    tuning: str,
    metrics: dict[str, float],
) -> str:
    train = np.unique(np.concatenate([tr for tr, _ in plan]))
    facts = flow_report.data_facts(
        data,
        train,
        flow_report.fold_hidden(data),
        train_rows=len(plan[0][0]),
        rows_label="sampled, per fold",
        note=f"One model per location fold ({len(plan)}): each trained on the other folds' "
        "sensors and scored on its own; the costs below are averaged over the folds.",
    )
    return flow_report.render(
        f"{family}_{data.variant}",
        f"Candidate `{family}` ({data.variant} features): "
        f"{FAMILY_NAMES.get(family, family)} with a Poisson objective.",
        facts,
        "Poisson deviance, 2 (y log(y / mu) - (y - mu)), for count y and prediction mu",
        "mean Poisson deviance over the rows (sampled, 20,000 per fold and set)",
        {"Cost by trees, mean over folds": flow_report.mean_curve(diag["curves"])},
        {
            f"{k} (mean per fold)": v
            for k, v in flow_report.mean_counts(diag["learned"]).items()
        },
        params,
        tuning,
        metrics,
        datetime.now(UTC).isoformat(timespec="seconds"),
    )


def train_variant(
    data: Data,
    fcfg: dict[str, Any],
    families: list[str],
    plan: list[tuple[np.ndarray, np.ndarray]],
    out: Path,
    params: dict[str, dict[str, Any]],
    baseline_preds: dict[str, np.ndarray],
    baseline_runs: dict[str, dict[str, Any]],
    plan_id: str,
    tuned: bool = False,
) -> dict[str, Any]:
    """Candidates for one variant, logged; writes oof.parquet, candidates.json and reports."""
    diag: dict[str, dict[str, list]] = {}
    preds, seconds = oof_candidates(data, families, params, plan, fcfg["seed"], diag)
    models: dict[str, Any] = {}
    for f in families:
        metrics = fold_scores(data, preds[f], plan, fcfg["calibration_days"])
        tuning = flow_report.tuning_text(tuned, fcfg["candidates"][f].get("search"))
        text = candidate_report(data, f, params[f], plan, diag[f], tuning, metrics)
        write_report(out, f"{f}_{data.variant}", text)
        run_id = log_run(
            f"{f}_{data.variant}",
            {**params[f], "variant": data.variant, "family": f},
            {
                "stoa.role": "candidate",
                "stoa.family": f,
                "stoa.variant": data.variant,
                "stoa.plan": plan_id,
            },
            {**metrics, "fit_s": seconds[f]},
            text,
        )
        models[f] = {"run_id": run_id, "params": params[f], "metrics": metrics}
        log.info(
            "%s_%s: log_mae_cv %.3f, wape_cv %.3f",
            f,
            data.variant,
            metrics["log_mae_cv"],
            metrics["wape_cv"],
        )

    out.mkdir(parents=True, exist_ok=True)
    tested = fold_rows(data, plan)
    oof = pd.DataFrame(
        {
            "row": tested,
            "location_id": data.sensors[tested],
            "ts_utc": data.ts.to_numpy()[tested],
            "fold": data.fold[tested],
            "y": data.y[tested].astype("float32"),
        }
    )
    for name, p in {**preds, **baseline_preds}.items():
        oof[name] = p[tested]
    oof.to_parquet(out / "oof.parquet", index=False)
    summary = {
        "variant": data.variant,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "plan": plan_id,
        "n_folds": len(plan),
        "train_rows": int(fcfg["train_rows_per_fold"]),
        "seed": int(fcfg["seed"]),
        "families": families,
        "tuned": tuned,
        "models": models,
        "baselines": baseline_runs,
    }
    (out / "candidates.json").write_text(json.dumps(summary, indent=2))
    return summary


def run(
    variants: list[str],
    families: list[str],
    fcfg: dict[str, Any],
    out_root: Path,
    load_data=flow_data.load,
    n_folds: int | None = None,
    max_trees: int | None = None,
    tune: bool = False,
    n_trials: int | None = None,
) -> dict[str, Any]:
    """Train every variant; returns each variant's candidates.json content."""
    seed, train_rows = fcfg["seed"], fcfg["train_rows_per_fold"]
    baseline_preds: dict[str, np.ndarray] | None = None
    baseline_runs: dict[str, dict[str, Any]] = {}
    reference: np.ndarray | None = None
    summaries = {}
    for variant in variants:
        t = time.perf_counter()
        data = load_data(variant)
        plan = flow_data.fold_plan(data.fold, train_rows, seed, n_folds)
        plan_id = plan_tag(len(plan), train_rows, seed)
        params = {
            f: candidates.capped(f, fcfg["candidates"][f]["params"], max_trees)
            for f in families
        }
        if tune:
            for f in families:
                tags = {
                    "stoa.role": "tuning",
                    "stoa.family": f,
                    "stoa.variant": variant,
                }
                with tracking.run(f"tune_{f}_{variant}", tags=tags):
                    best = flow_tune.tune(data, f, fcfg, n_trials, max_trees)
                params[f] = candidates.capped(f, best, max_trees)

        # Both variants have the same rows; the baselines use network features only.
        if baseline_preds is None:
            reference = data.sensors
            fitted: dict[str, list] = {}
            baseline_preds = oof_baselines(data, plan, fitted)
            for name, p in baseline_preds.items():
                metrics = fold_scores(data, p, plan, fcfg["calibration_days"])
                text = baseline_report(data, name, plan, fitted[name], metrics)
                write_report(out_root / "baselines", name, text)
                bparams = {
                    "rung": name.removeprefix("baseline_"),
                    "ridge_alpha": "leave-one-sensor-out",
                }
                run_id = log_run(
                    name,
                    bparams,
                    {
                        "stoa.role": "baseline",
                        "stoa.family": name,
                        "stoa.variant": "network",
                        "stoa.plan": plan_id,
                    },
                    metrics,
                    text,
                )
                baseline_runs[name] = {"run_id": run_id, "metrics": metrics}
                log.info("%s: log_mae_cv %.3f", name, metrics["log_mae_cv"])
        elif not np.array_equal(reference, data.sensors):
            raise ValueError("the variants' training frames have different rows")

        summaries[variant] = train_variant(
            data,
            fcfg,
            families,
            plan,
            out_root / variant,
            params,
            baseline_preds,
            baseline_runs,
            plan_id,
            tuned=tune,
        )
        log.info("%s: %.0f s", variant, time.perf_counter() - t)
        del data
    return summaries


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", choices=["rich", "portable", "both"], default="both")
    ap.add_argument("--families", nargs="+", choices=candidates.FAMILIES)
    ap.add_argument(
        "--tune", action="store_true", help="Optuna search per family first"
    )
    ap.add_argument("--n-trials", type=int)
    ap.add_argument("--folds", type=int, help="use only the first N folds (quick runs)")
    ap.add_argument("--max-trees", type=int, help="cap trees per model (quick runs)")
    ap.add_argument("--train-rows", type=int, help="training rows per fold")
    ap.add_argument("--output-dir")
    args = ap.parse_args()

    fcfg = flow_data.flow_config()
    if args.train_rows:
        fcfg["train_rows_per_fold"] = args.train_rows
    out = flow_data.output_dir(fcfg, args.output_dir)
    variants = fcfg["variants"] if args.variant == "both" else [args.variant]
    families = args.families or list(fcfg["candidates"])
    run(
        variants,
        families,
        fcfg,
        out,
        n_folds=args.folds,
        max_trees=args.max_trees,
        tune=args.tune,
        n_trials=args.n_trials,
    )


if __name__ == "__main__":
    main()
