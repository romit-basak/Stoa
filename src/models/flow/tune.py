"""Optuna search over one candidate family's hyperparameters (TPE sampler).

Trials are scored by mae_log on inner folds: the fold sensors split again by their spatial
group, on a smaller row sample. Holdout precincts never enter. Tuned and untuned runs are scored
on the same outer folds, which the inner folds also draw from, so tuned *_cv metrics are a
little optimistic; the default (untuned) parameters in configs/models.yaml avoid that.

Each trial is a nested MLflow run under the parent `tune_<family>_<variant>`, and the parent
logs the best parameters and Optuna's estimate of how much each one mattered (the
hyperparameter sensitivity the course asks for).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

from src.models.flow import candidates
from src.models.flow.data import Data
from src.models.flow.metrics import mae_log


def suggest(trial: Any, space: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "float_log":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=True
            )
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif kind == "choice":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"{name}: unknown search type {kind!r}")
    return params


def inner_plan(
    data: Data, n_folds: int, train_rows: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Inner folds over the fold sensors, split by spatial group, each side sampled."""
    rng = np.random.default_rng(seed)
    rows = np.flatnonzero(data.fold >= 0)
    plan = []
    folds = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train, test in folds.split(rows, groups=data.group[rows]):
        train, test = rows[train], rows[test]
        if len(train) > train_rows:
            train = np.sort(rng.choice(train, train_rows, replace=False))
        if len(test) > train_rows:
            test = np.sort(rng.choice(test, train_rows, replace=False))
        plan.append((train, test))
    return plan


def tune(
    data: Data,
    family: str,
    fcfg: dict[str, Any],
    n_trials: int | None = None,
    max_trees: int | None = None,
) -> dict[str, Any]:
    """Best parameters for `family` (defaults overlaid with the best trial's values)."""
    import mlflow
    import optuna

    tcfg = fcfg["tune"]
    spec = fcfg["candidates"][family]
    plan = inner_plan(data, tcfg["inner_folds"], tcfg["train_rows"], fcfg["seed"])
    built = [(data.frame.matrix(tr), data.frame.matrix(te), tr, te) for tr, te in plan]
    y = data.y

    def objective(trial: Any) -> float:
        params = candidates.capped(
            family, {**spec["params"], **suggest(trial, spec["search"])}, max_trees
        )
        losses = []
        for x_tr, x_te, tr, te in built:
            model = candidates.make(family, params, fcfg["seed"]).fit(x_tr, y[tr])
            losses.append(mae_log(y[te], candidates.predict(model, x_te)))
        loss = float(np.mean(losses))
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("mae_log_inner", loss)
        return loss

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=fcfg["seed"])
    )
    study.optimize(objective, n_trials=n_trials or tcfg["n_trials"])
    best = {**spec["params"], **study.best_params}
    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    mlflow.log_metric("mae_log_inner_best", study.best_value)
    if len(study.trials) > 1:
        try:
            importance = optuna.importance.get_param_importances(study)
            mlflow.log_dict(importance, "param_importance.json")
        except (ValueError, RuntimeError):
            pass  # too few distinct trials to estimate importances
    return best
