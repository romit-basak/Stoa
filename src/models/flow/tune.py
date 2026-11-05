"""Hyperparameter search for the flow model using Optuna TPE.

Usage: python -m src.models.flow.tune [--variant rich|portable|both] [--n-trials 50]
       [--if-older-than DAYS]

Each trial trains a Poisson HistGradientBoostingRegressor on a 200k-row subsample
of the training partition (inner 2-fold GroupKFold). Objective is mean WAPE.
Each trial is logged as a nested MLflow child run so the full search is auditable.

The best params are written to configs/model.yaml under `best_params`, with a
`tuned_at` timestamp so the --if-older-than guard can skip fresh searches.
"""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor

from src.features.training_frame import VARIANTS, fold_plan, load
from src.ingest.common import REPO_ROOT, setup_logging
from src.models.flow.evaluate import wape
from src.models.flow.features import design_matrix, sensor_features

MODEL_YAML = REPO_ROOT / "configs" / "model.yaml"


def _sample_params(trial: Any, space: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in space.items():
        t = spec["type"]
        if t == "float_log":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        elif t == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif t == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif t == "int_choice" or t == "float_choice":
            params[name] = trial.suggest_categorical(name, spec["choices"])
    return params


def build_study(
    hourly: pd.DataFrame,
    osm: pd.DataFrame,
    cfg: dict[str, Any],
    variant: str,
    n_trials: int = 50,
    inner_folds: int = 2,
) -> dict[str, Any]:
    """Optuna TPE search. Returns the best params dict found."""
    import mlflow
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tcfg = cfg["training"]
    space = cfg["search_space"]
    tune_cfg = cfg["tune"]
    subsample = tune_cfg["subsample_rows"]
    seed = tcfg["seed"]

    sensors = hourly["location_id"].to_numpy()
    y = hourly["pedestriancount"].to_numpy(dtype=float)
    station_cap_m = 2000.0  # matches built_form.max_distance_m in features.yaml

    # Build the full feature matrix once; subsample per trial
    from src.features.training_frame import derived_features, row_features
    r_cols = row_features(variant)
    d_cols = derived_features(variant)
    all_row_cols = r_cols + d_cols
    osm_parquet = (
        REPO_ROOT / "data" / "processed" / "features" / "melbourne" / "sensor_osm_features.parquet"
    )
    osm_sf = sensor_features(pd.read_parquet(osm_parquet), variant)

    plan = fold_plan(sensors, inner_folds, subsample, seed)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: Any) -> float:
        params = _sample_params(trial, space)
        fold_wapes = []
        for train_pos, test_pos in plan:
            x_train = np.nan_to_num(
                design_matrix(hourly, train_pos, osm_sf, station_cap_m, all_row_cols),
                nan=0.0,
            )
            x_test = np.nan_to_num(
                design_matrix(hourly, test_pos, osm_sf, station_cap_m, all_row_cols),
                nan=0.0,
            )
            model = HistGradientBoostingRegressor(
                loss="poisson",
                random_state=seed,
                early_stopping=False,
                **{k: v for k, v in params.items() if k != "early_stopping"},
            )
            model.fit(x_train, y[train_pos])
            pred = np.maximum(model.predict(x_test), 0.0)
            fold_wapes.append(wape(y[test_pos], pred))

        mean_wape = float(np.mean(fold_wapes))

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            mlflow.log_params(params)
            mlflow.log_metric("wape_inner_cv", mean_wape)
        return mean_wape

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _is_recent(cfg: dict[str, Any], variant: str, days: float, now: datetime) -> bool:
    bp = cfg.get("best_params", {}).get(variant)
    tuned_at = cfg.get("tuned_at", {}).get(variant)
    if not bp or not tuned_at:
        return False
    return now - datetime.fromisoformat(tuned_at) < timedelta(days=days)


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=[*VARIANTS, "both"], default="both")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--if-older-than", type=float, metavar="DAYS")
    args = ap.parse_args()

    import mlflow

    cfg = yaml.safe_load(MODEL_YAML.read_text())
    now = datetime.now(UTC)
    variants = list(VARIANTS) if args.variant == "both" else [args.variant]

    if args.if_older_than is not None:
        variants = [
            v for v in variants if not _is_recent(cfg, v, args.if_older_than, now)
        ]
        if not variants:
            print(f"Hyperparameters are under {args.if_older_than} days old; skipped.")
            return

    mlflow.set_tracking_uri(f"file:{REPO_ROOT / 'mlruns'}")
    mlflow.set_experiment("stoa_flow_model_tuning")

    if "best_params" not in cfg:
        cfg["best_params"] = {}
    if "tuned_at" not in cfg:
        cfg["tuned_at"] = {}

    for variant in variants:
        frame = load(variant=variant)
        hourly_eligible = frame.hourly
        osm = frame.osm

        t = time.perf_counter()
        with mlflow.start_run(run_name=f"tune_{variant}"):
            best = build_study(
                hourly_eligible, osm, cfg, variant, args.n_trials, cfg["tune"]["inner_folds"]
            )
        print(
            f"{variant}: best params {best} in {time.perf_counter() - t:.0f} s"
        )
        cfg["best_params"][variant] = best
        cfg["tuned_at"][variant] = now.isoformat()

    MODEL_YAML.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(f"Wrote best params to {MODEL_YAML}")


if __name__ == "__main__":
    main()
