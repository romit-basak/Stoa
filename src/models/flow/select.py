"""Pick each variant's flow model after the bias check, fit it, and evaluate it.

Usage: python -m src.models.flow.select [--variant rich|portable|both] [--output-dir DIR]
       [--final-train-rows N]

Reads <output_dir>/<variant>/oof.parquet and candidates.json from src.models.flow.train, then:
1. ranks the candidates by log_mae_cv (the success criterion);
2. for those within `selection.tolerance` of the best, measures WAPE per slice of their
   out-of-fold predictions with the bias slices of configs/bias.yaml (src/bias: Fairlearn
   MetricFrame, sensor-bootstrap intervals). The candidate whose worst slice is least worse
   than its overall error wins, so a small accuracy gain can't buy a model that fails one
   group;
3. fits the winner's 10% and 90% quantile models on the same folds and widens their interval
   from the out-of-fold misses (`interval_margins`), for coverage80_cv;
4. fits the winner and its interval models on every fold sensor, and scores the holdout
   precincts (never seen until now), the event days, other days and eras
   (src.models.flow.evaluate.standard_parts), the calibration curve and the feature
   importance;
5. logs it as `<family>_<variant>` with stoa.role = selected: the three models, the ranking,
   the slice table, the comparison charts and the training report (src/models/flow/report.py,
   also at <output_dir>/<variant>/reports/selected_<family>_<variant>.md).

Writes <output_dir>/<variant>/selected.json for src.models.flow.registry.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from src.features.common import load_features_config, log
from src.ingest.common import REPO_ROOT, load_config, setup_logging
from src.models import tracking
from src.models.flow import candidates
from src.models.flow import data as flow_data
from src.models.flow import report as flow_report
from src.models.flow.data import Data, calibration_window
from src.models.flow.evaluate import (
    cv_coverage,
    log_mae,
    score,
    score_parts,
    scored_mask,
    standard_parts,
)

# --------------------------------------------------------------------------------------------
# Ranking and the bias check
# --------------------------------------------------------------------------------------------


def ranking(summary: dict[str, Any]) -> pd.DataFrame:
    """Candidates and baselines with their fold metrics, best log_mae_cv first."""
    rows = []
    for kind, models in (
        ("candidate", summary["models"]),
        ("baseline", summary["baselines"]),
    ):
        for name, m in models.items():
            rows.append({"model": name, "kind": kind, **m["metrics"]})
    return pd.DataFrame(rows).sort_values("log_mae_cv").reset_index(drop=True)


def shortlist(rank: pd.DataFrame, tolerance: float) -> list[str]:
    cand = rank[rank["kind"] == "candidate"]
    best = float(cand["log_mae_cv"].min())
    return cand.loc[cand["log_mae_cv"] <= best * (1 + tolerance), "model"].tolist()


def slice_rows(oof: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """The bias slices for each out-of-fold row, built by src/bias for the whole table."""
    from src.bias.representation import sliced_rows
    from src.bias.slices import load_bias_config

    bias_cfg = load_bias_config()
    sl = sliced_rows(load_config(), load_features_config(), bias_cfg)
    rows = sl["rows"].assign(ts_utc=sl["hourly"]["ts_utc"].to_numpy())
    keys = oof[["location_id", "ts_utc"]]
    aligned = keys.merge(rows, on=["location_id", "ts_utc"], how="left", validate="1:1")
    return aligned, sl["dims"], bias_cfg


def worst_slice(errors: pd.DataFrame, min_sensors: int) -> tuple[float, str]:
    """Largest group WAPE over overall WAPE among groups with at least min_sensors sensors."""
    ok = errors[errors["n_sensors"] >= min_sensors]
    if ok.empty:
        return float("nan"), ""
    top = ok.loc[ok["wape_vs_overall"].idxmax()]
    return float(top["wape_vs_overall"]), f"{top['dimension']}={top['group']}"


def bias_check(
    oof: pd.DataFrame,
    names: list[str],
    skip: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-slice errors for each named model, and each one's worst slice."""
    from src.bias.slice_errors import slice_errors

    rows, dims, bias_cfg = slice_rows(oof)
    dims = [d for d in dims if d not in skip]
    y = oof["y"].to_numpy(dtype=float)
    tables, worst = [], []
    for name in names:
        t = slice_errors(rows, dims, y, oof[name].to_numpy(dtype=float), bias_cfg)
        t.insert(0, "model", name)
        tables.append(t)
        ratio, group = worst_slice(t, bias_cfg["thin_min_sensors"])
        worst.append(
            {
                "model": name,
                "worst_slice_ratio": ratio,
                "worst_slice": group,
                "groups_worse": int(t["worse"].sum()),
            }
        )
    return pd.concat(tables, ignore_index=True), pd.DataFrame(worst)


def choose(rank: pd.DataFrame, worst: pd.DataFrame) -> str:
    """The shortlisted candidate with the smallest worst-slice ratio (ties: lower log MAE)."""
    merged = worst.merge(rank[["model", "log_mae_cv"]], on="model")
    return str(merged.sort_values(["worst_slice_ratio", "log_mae_cv"]).iloc[0]["model"])


# --------------------------------------------------------------------------------------------
# Fitting and evaluating the winner
# --------------------------------------------------------------------------------------------


def oof_intervals(
    data: Data,
    family: str,
    params: dict[str, Any],
    plan: list[tuple[np.ndarray, np.ndarray]],
    quantiles: list[float],
    seed: int,
) -> list[np.ndarray]:
    preds = [np.full(len(data.y), np.nan, dtype="float32") for _ in quantiles]
    for train, test in plan:
        x_train, x_test = data.frame.matrix(train), data.frame.matrix(test)
        for p, q in zip(preds, quantiles):
            model = candidates.fit(
                candidates.make(family, params, seed, quantile=q),
                x_train,
                data.y[train],
            )
            p[test] = candidates.predict(model, x_test)
    return preds


def interval_margins(
    y: np.ndarray, low: np.ndarray, high: np.ndarray, quantiles: list[float]
) -> tuple[float, float]:
    """How far to widen each side of the quantile interval (conformalized quantile regression).

    The 10% and 90% models learn the spread of counts at the sensors they trained on, but at a
    new location most of the error is in the sensor's overall level, so their interval is too
    narrow there. On log1p counts (level errors are multiplicative), the lower margin is the
    quantile of how far counts fall below `low` that leaves `quantiles[0]` of them below it
    after widening, and likewise above `high`. A margin is negative when that side is too wide.
    """
    ly, llo, lhi = np.log1p(y), np.log1p(low), np.log1p(high)
    lower = float(np.quantile(llo - ly, 1 - quantiles[0]))
    upper = float(np.quantile(ly - lhi, quantiles[1]))
    return lower, upper


def widen(
    low: np.ndarray, high: np.ndarray, margins: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = margins
    low = np.maximum(np.expm1(np.log1p(low) - lower), 0.0)
    return low, np.expm1(np.log1p(high) + upper)


def cross_fold_widen(
    y: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    folds: np.ndarray,
    quantiles: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Each fold's interval widened by margins from the other folds' rows only, so the
    coverage of the result is measured on sensors the margins never saw."""
    new_low, new_high = np.empty(len(y)), np.empty(len(y))
    for k in np.unique(folds):
        m = folds == k
        margins = interval_margins(y[~m], low[~m], high[~m], quantiles)
        new_low[m], new_high[m] = widen(low[m], high[m], margins)
    return new_low, new_high


def fit_final(
    data: Data,
    family: str,
    params: dict[str, Any],
    quantiles: list[float],
    rows: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    x = data.frame.matrix(rows)
    y = data.y[rows]
    models = {"model": candidates.fit(candidates.make(family, params, seed), x, y)}
    for q in quantiles:
        models[f"model_q{round(q * 100):02d}"] = candidates.fit(
            candidates.make(family, params, seed, quantile=q), x, y
        )
    return models


def holdout_scores(
    data: Data, model: Any, calibration_days: float
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Each holdout precinct's error, uncalibrated and calibrated on 1-2 of its sensors.

    Calibrating on k sensors: one scale from their first calibration_days of counts, applied
    to the precinct's other sensors and scored on their later hours, averaged over every choice
    of k sensors. The UniMelb edge is split at the Metro Tunnel opening (CLAUDE.md).
    Returns the table and the uncalibrated scores of every holdout precinct row after the
    calibration windows (`log_mae_holdout`, `wape_holdout`).
    """
    named = (data.holdout != "") & (data.holdout != "neighbour")
    idx = np.flatnonzero(named)
    if not len(idx):
        return pd.DataFrame(), {}
    pred = candidates.predict(model, data.frame.matrix(idx))
    rows = data.frame.hourly.iloc[idx][["location_id"]].reset_index(drop=True)
    y, sensors, precinct = data.y[idx], data.sensors[idx], data.holdout[idx]
    window = calibration_window(sensors, data.ts.iloc[idx], calibration_days)
    post = data.frame.hourly["post_metro_tunnel"].to_numpy(dtype=bool)[idx]
    out = []
    for name in sorted(set(precinct)):
        in_p = precinct == name
        parts = {"all": in_p}
        if name == "unimelb_edge":
            parts = {"pre_metro_tunnel": in_p & ~post, "post_metro_tunnel": in_p & post}
        for part, mask in parts.items():
            ids = np.unique(sensors[mask])
            for k in (0, 1, 2):
                if k >= len(ids):
                    continue
                scores = []
                for chosen in itertools.combinations(ids, k):
                    fit = mask & np.isin(sensors, chosen) & window
                    p_sum = float(pred[fit].sum())
                    scale = float(y[fit].sum()) / p_sum if k and p_sum > 0 else 1.0
                    m = mask & ~np.isin(sensors, chosen) & ~window
                    if m.any():
                        scores.append(score(rows[m], y[m], scale * pred[m]))
                if scores:
                    out.append(
                        {
                            "precinct": name,
                            "period": part,
                            "calibrated_on": k,
                            "sensors": len(ids),
                            "log_mae": float(np.mean([s["log_mae"] for s in scores])),
                            "wape": float(np.mean([s["wape"] for s in scores])),
                        }
                    )
    s = score(rows[~window], y[~window], pred[~window])
    return pd.DataFrame(out), {
        "log_mae_holdout": s["log_mae"],
        "wape_holdout": s["wape"],
    }


def part_scores(
    data: Data, oof: pd.DataFrame, pred: np.ndarray, calibration_days: float
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Out-of-fold scores on event days, other days and each era, reported apart (CLAUDE.md).

    The parts come from src.models.flow.evaluate.standard_parts, with the eras of
    configs/bias.yaml and the event days the training frame lists, on the hours after each
    sensor's calibration window. Returns one row per part and the metrics to log, named
    `log_mae_<part>` and `wape_<part>`.
    """
    from src.bias.slices import load_bias_config

    path = (
        REPO_ROOT
        / load_features_config()["processed_dir"]
        / "training"
        / data.site
        / "event_days.parquet"
    )
    event_dates = (
        pd.read_parquet(path)["date_local"]
        if path.exists()
        else pd.Series([], dtype="datetime64[us]")
    )
    tested = oof["row"].to_numpy()
    rows = data.frame.hourly.iloc[tested][["location_id", "ts_utc", "date_local"]]
    rows = rows.reset_index(drop=True)
    keep = scored_mask(rows["location_id"].to_numpy(), rows["ts_utc"], calibration_days)
    rows = rows[keep].reset_index(drop=True)
    y = oof["y"].to_numpy(dtype=float)[keep]
    parts = standard_parts(rows, event_dates, load_bias_config()["eras"])
    scored = score_parts(rows, y, pred[keep], parts)
    table = pd.DataFrame([{"part": name, **s} for name, s in scored.items()])
    metrics = {
        f"{m}_{name}": float(s[m])
        for name, s in scored.items()
        if s["rows"]
        for m in ("log_mae", "wape")
    }
    return table, metrics


def calibration_curve(
    rows: pd.DataFrame,
    y: np.ndarray,
    pred: np.ndarray,
    calibration_days: float,
    ks: list[int],
    reps: int,
    seed: int,
) -> pd.DataFrame:
    """Error on the other sensors after scaling the model to k sensors' first weeks of counts.

    One scale for the whole site, from the k sensors' calibration windows, applied to every
    other sensor and scored with src.models.flow.evaluate.score on their hours after the
    window. k = 0 is the uncalibrated model. `rows` (location_id, ts_utc) are out-of-fold, so
    no sensor is scored by a model that trained on it.
    """
    rng = np.random.default_rng(seed)
    sensors = rows["location_id"].to_numpy()
    window = calibration_window(sensors, rows["ts_utc"], calibration_days)
    ids = np.unique(sensors)
    out = []
    for k in [0, *ks]:
        if k >= len(ids):
            continue
        for rep in range(reps if k else 1):
            chosen = rng.choice(ids, size=k, replace=False) if k else ids[:0]
            fit = np.isin(sensors, chosen) & window
            p_sum = float(pred[fit].sum())
            scale = float(y[fit].sum()) / p_sum if k and p_sum > 0 else 1.0
            m = ~np.isin(sensors, chosen) & ~window
            s = score(rows[m], y[m], scale * pred[m])
            out.append(
                {
                    "k": k,
                    "rep": rep,
                    "scale": scale,
                    "log_mae": s["log_mae"],
                    "wape": s["wape"],
                }
            )
    return pd.DataFrame(out)


def importance(
    model: Any, x: np.ndarray, y: np.ndarray, names: list[str], seed: int
) -> tuple[pd.DataFrame, str]:
    """Mean |SHAP| per feature (log-scale for Poisson boosters), else permutation importance."""
    try:
        import shap

        values = shap.TreeExplainer(model).shap_values(x)
        imp = np.abs(np.asarray(values)).mean(axis=0)
        method = "shap"
    except Exception as e:  # noqa: BLE001 - any SHAP failure falls back to permutation
        from sklearn.inspection import permutation_importance
        from sklearn.metrics import make_scorer

        log.info(
            "SHAP unavailable for this model (%s); using permutation importance", e
        )

        def neg_log_mae(yt: np.ndarray, yp: np.ndarray) -> float:
            return -log_mae(yt, np.maximum(yp, 0.0))

        res = permutation_importance(
            model,
            x,
            y,
            scoring=make_scorer(neg_log_mae),
            n_repeats=3,
            random_state=seed,
        )
        imp, method = res.importances_mean, "permutation"
    table = pd.DataFrame({"feature": names, "importance": imp})
    return table.sort_values("importance", ascending=False).reset_index(
        drop=True
    ), method


# --------------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------------


def comparison_chart(
    rank: pd.DataFrame, worst: pd.DataFrame, winner: str, path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5))
    order = rank.sort_values("log_mae_cv", ascending=False)
    colors = [
        "#c0392b" if m == winner else ("#95a5a6" if k == "baseline" else "#2e86c1")
        for m, k in zip(order["model"], order["kind"])
    ]
    a.barh(order["model"], order["log_mae_cv"], color=colors)
    a.set_xlabel("MAE on log counts, mean over location folds (lower is better)")
    a.set_title("Candidates (blue), baselines (grey), selected (red)")
    w = worst.sort_values("worst_slice_ratio", ascending=False)
    b.barh(
        w["model"],
        w["worst_slice_ratio"],
        color=["#c0392b" if m == winner else "#2e86c1" for m in w["model"]],
    )
    b.axvline(1.0, color="#555", linewidth=0.8)
    b.set_xlabel("Worst slice WAPE / overall WAPE (lower is fairer)")
    b.set_title("Shortlisted candidates: bias check")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def curve_chart(curve: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = curve.groupby("k")["wape"]
    med, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(med.index, med.to_numpy(), marker="o")
    ax.fill_between(med.index, lo.to_numpy(), hi.to_numpy(), alpha=0.25)
    ax.set_xlabel("Sensors calibrated on (k)")
    ax.set_ylabel("WAPE on the other sensors (median, IQR)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def importance_chart(table: pd.DataFrame, method: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = table.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel(
        "mean |SHAP|" if method == "shap" else "permutation importance (log MAE)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------------------------


def selected_report(
    data: Data,
    family: str,
    params: dict[str, Any],
    quantiles: list[float],
    models: dict[str, Any],
    rows: np.ndarray,
    tuning: str,
    metrics: dict[str, float],
    seed: int,
) -> str:
    """The training report of the final model and its interval models."""
    from datetime import UTC, datetime

    held = np.flatnonzero((data.holdout != "") & (data.holdout != "neighbour"))
    tr = flow_report.sample(rows, seed=seed)
    sets = {"training rows": (data.frame.matrix(tr), data.y[tr])}
    if len(held):
        he = flow_report.sample(held, seed=seed)
        sets["holdout precincts"] = (data.frame.matrix(he), data.y[he])
    y_train = data.y[rows]
    curves = {
        "Mean model (Poisson deviance)": flow_report.cost_curve(
            family, models["model"], sets, float(np.mean(y_train))
        )
    }
    for q in quantiles:
        name = f"model_q{round(q * 100):02d}"
        curves[f"{q:.0%} interval model (pinball loss)"] = flow_report.cost_curve(
            family, models[name], sets, float(np.quantile(y_train, q)), quantile=q
        )
    learned = {
        f"{name}: {k.replace('_', ' ')}": v
        for name, m in models.items()
        for k, v in flow_report.learned(family, m).items()
    }
    facts = flow_report.data_facts(
        data,
        rows,
        {"holdout precincts and their neighbours": data.sensors[data.fold < 0]},
        note="Trained on every fold sensor; the holdout precincts are scored here for the "
        "first time.",
    )
    return flow_report.render(
        f"selected: {family}_{data.variant}",
        f"The selected flow model for the {data.variant} variant: `{family}` with a Poisson "
        f"objective, and `{family}` quantile models at {', '.join(f'{q:g}' for q in quantiles)} "
        "for the interval (widened by `interval_margins` in `selected.json`).",
        facts,
        "Poisson deviance for the mean model; pinball loss for the interval models",
        "the mean loss over the rows (sampled, 20,000 per set)",
        curves,
        learned,
        params,
        tuning,
        metrics,
        datetime.now(UTC).isoformat(timespec="seconds"),
    )


def log_model(model: Any, family: str, name: str, x_example: np.ndarray) -> str:
    if family == "lgbm":
        import mlflow.lightgbm

        info = mlflow.lightgbm.log_model(model, name=name, input_example=x_example)
    elif family == "xgb":
        import mlflow.xgboost

        info = mlflow.xgboost.log_model(model, name=name, input_example=x_example)
    else:
        info = tracking.log_sklearn_model(model, name=name, input_example=x_example)
    return info.model_uri


def select_variant(
    data: Data,
    fcfg: dict[str, Any],
    out: Path,
    final_train_rows: int | None = None,
    bias_fn=bias_check,
) -> dict[str, Any]:
    summary = json.loads((out / "candidates.json").read_text())
    oof = pd.read_parquet(out / "oof.parquet")
    scfg, seed = fcfg["selection"], fcfg["seed"]
    rank = ranking(summary)
    short = shortlist(rank, scfg["tolerance"])
    slices, worst = bias_fn(oof, short, scfg["skip_dimensions"])
    winner = choose(rank, worst)
    log.info("%s: shortlisted %s, selected %s", data.variant, short, winner)

    params = summary["models"][winner]["params"]
    # The same plan as train: its settings are in candidates.json.
    plan = flow_data.fold_plan(
        data.fold, summary["train_rows"], summary["seed"], summary["n_folds"]
    )
    if not np.array_equal(data.sensors[oof["row"].to_numpy()], oof["location_id"]):
        raise ValueError("oof.parquet doesn't match the training frame; rerun train")
    q = scfg["intervals"]
    low, high = oof_intervals(data, winner, params, plan, q, seed)
    tested = oof["row"].to_numpy()
    oof_rows = data.frame.hourly.iloc[tested][["location_id", "ts_utc"]]
    oof_rows = oof_rows.reset_index(drop=True)
    keep = scored_mask(
        oof_rows["location_id"].to_numpy(), oof_rows["ts_utc"], fcfg["calibration_days"]
    )
    scored = tested[keep]
    y_s, low_s, high_s, folds_s = (
        data.y[scored],
        low[scored],
        high[scored],
        data.fold[scored],
    )
    margins = interval_margins(y_s, low_s, high_s, q)
    wide_low, wide_high = cross_fold_widen(y_s, low_s, high_s, folds_s, q)
    cover = {
        **cv_coverage(y_s, low_s, high_s, folds_s, name="coverage80_raw"),
        **cv_coverage(y_s, wide_low, wide_high, folds_s),
        "interval_margin_low": margins[0],
        "interval_margin_high": margins[1],
        "interval_width_ratio": float(np.median((1 + wide_high) / (1 + wide_low))),
    }

    rows = flow_data.final_rows(
        data.fold, final_train_rows or fcfg["final_train_rows"], seed
    )
    models = fit_final(data, winner, params, q, rows, seed)
    precincts, holdout = holdout_scores(data, models["model"], fcfg["calibration_days"])
    pred = oof[winner].to_numpy(dtype=float)
    parts, part_metrics = part_scores(data, oof, pred, fcfg["calibration_days"])
    ccfg = fcfg["calibration_curve"]
    curve = calibration_curve(
        oof_rows,
        oof["y"].to_numpy(dtype=float),
        pred,
        fcfg["calibration_days"],
        ccfg["k"],
        ccfg["reps"],
        seed,
    )
    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(rows, size=min(2000, len(rows)), replace=False))
    x_sample = data.frame.matrix(sample)
    imp, method = importance(
        models["model"], x_sample, data.y[sample], data.frame.columns, seed
    )

    w = worst.set_index("model").loc[winner]
    metrics = {
        **summary["models"][winner]["metrics"],
        **cover,
        **holdout,
        "worst_slice_ratio": float(w["worst_slice_ratio"]),
        "groups_worse": int(w["groups_worse"]),
        **part_metrics,
    }
    art = out / "selected"
    art.mkdir(parents=True, exist_ok=True)
    tuning = flow_report.tuning_text(
        summary.get("tuned", False), fcfg["candidates"][winner].get("search")
    )
    text = selected_report(data, winner, params, q, models, rows, tuning, metrics, seed)
    (art / "model_report.md").write_text(text)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    (out / "reports" / f"selected_{winner}_{data.variant}.md").write_text(text)
    rank.to_csv(art / "ranking.csv", index=False)
    slices.to_csv(art / "slices.csv", index=False)
    worst.to_csv(art / "bias_check.csv", index=False)
    precincts.to_csv(art / "holdout_precincts.csv", index=False)
    parts.to_csv(art / "parts.csv", index=False)
    curve.to_csv(art / "calibration_curve.csv", index=False)
    imp.to_csv(art / "feature_importance.csv", index=False)
    comparison_chart(rank, worst, winner, art / "model_comparison.png")
    curve_chart(curve, art / "calibration_curve.png")
    importance_chart(imp, method, art / "feature_importance.png")

    tags = {
        "stoa.role": "selected",
        "stoa.family": winner,
        "stoa.variant": data.variant,
        "stoa.plan": summary["plan"],
        "stoa.candidate_run": summary["models"][winner]["run_id"],
        "stoa.worst_slice": str(w["worst_slice"]),
        "stoa.importance": method,
    }
    run_params = {
        **params,
        "variant": data.variant,
        "family": winner,
        "final_train_rows": len(rows),
        "shortlist": ",".join(short),
    }
    with tracking.run(
        f"{winner}_{data.variant}", params=run_params, tags=tags
    ) as active:
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(art), artifact_path="selection")
        mlflow.log_text(text, "model_report.md")
        mlflow.log_dict(data.frame.columns, "features.json")
        example = x_sample[:5]
        uris = {name: log_model(m, winner, name, example) for name, m in models.items()}
        mlflow.set_tag("stoa.model_uri", uris["model"])
        run_id = active.info.run_id

    result = {
        "variant": data.variant,
        "run_id": run_id,
        "family": winner,
        "model_uris": uris,
        # Applied to model_q10 and model_q90 at prediction time with `widen`.
        "interval_margins": {"low": margins[0], "high": margins[1]},
        "metrics": metrics,
        "baseline_space_syntax": summary["baselines"]
        .get("baseline_space_syntax", {})
        .get("metrics", {}),
    }
    (out / "selected.json").write_text(json.dumps(result, indent=2))
    log.info(
        "%s: selected %s, log_mae_cv %.3f, coverage80_cv %.3f, wape_holdout %.3f",
        data.variant,
        winner,
        metrics["log_mae_cv"],
        metrics["coverage80_cv"],
        metrics.get("wape_holdout", float("nan")),
    )
    return result


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", choices=["rich", "portable", "both"], default="both")
    ap.add_argument("--output-dir")
    ap.add_argument("--final-train-rows", type=int)
    args = ap.parse_args()
    fcfg = flow_data.flow_config()
    out = flow_data.output_dir(fcfg, args.output_dir)
    variants = fcfg["variants"] if args.variant == "both" else [args.variant]
    for v in variants:
        select_variant(flow_data.load(v), fcfg, out / v, args.final_train_rows)


if __name__ == "__main__":
    main()
