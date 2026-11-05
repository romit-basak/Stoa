"""Train the Melbourne flow model: Melbourne-rich and portable variants.

Usage: python -m src.models.flow.train [--variant rich|portable|both] [--tune]
       [--if-older-than DAYS]

For each variant:
- Grouped K-fold OOF predictions (Poisson loss + quantile models for 80% intervals)
- MLflow experiment tracking with metrics, params, and artefacts
- OOF predictions saved to data/processed/models/flow/{variant}/oof_predictions.parquet
- Final model saved to data/processed/models/flow/{variant}/model.pkl

The --tune flag runs Optuna search before training (see src/models/flow/tune.py).
The --if-older-than DAYS guard skips when a recent model exists with the same settings.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor

from src.features.common import load_features_config
from src.features.training_frame import VARIANTS, fold_plan, load
from src.ingest.common import REPO_ROOT, load_config, setup_logging
from src.models.flow.baselines import run_baselines
from src.models.flow.evaluate import (
    baseline_ladder,
    calibration_curve,
    check_gates,
    interval_coverage,
    mae_log,
    shap_summary,
    transfer_precinct_eval,
    wape,
)
from src.models.flow.features import (
    calibration_window,
    design_matrix,
    feature_names,
)

MODEL_YAML = REPO_ROOT / "configs" / "model.yaml"
BIAS_YAML = REPO_ROOT / "configs" / "bias.yaml"


def _load_configs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model_cfg = yaml.safe_load(MODEL_YAML.read_text())
    data_cfg = load_config()
    return model_cfg, data_cfg, load_features_config()


def _out_dir(variant: str) -> Path:
    return REPO_ROOT / "data" / "processed" / "models" / "flow" / variant


def _model_params(cfg: dict[str, Any], variant: str) -> dict[str, Any]:
    best = cfg.get("best_params", {}).get(variant)
    return best if best else cfg["default_params"]


def run_variant(
    hourly: pd.DataFrame,
    osm: pd.DataFrame,
    cfg: dict[str, Any],
    variant: str,
    n_iter: int | None = None,
    folds: int | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Train one variant and return a results dict.

    Exposed as a function (not just CLI) so the smoke tests can call it
    without subprocess overhead and without a real MLflow server.
    """
    tcfg = cfg["training"]
    station_cap_m = 2000.0
    cal_days = tcfg["calibration_days"]

    from src.features.training_frame import derived_features, row_features
    r_cols = row_features(variant)
    d_cols = derived_features(variant)
    all_row_cols = [c for c in r_cols + d_cols if c in hourly.columns]
    feat_names = feature_names(osm, all_row_cols)

    # `osm` is expected to be sensor_features() output: indexed by location_id, float32.
    # If it's not yet indexed, index it here so both paths work.
    if "location_id" in osm.columns:
        osm = osm.set_index("location_id").astype("float32")
    elif osm.index.name != "location_id":
        osm = osm.astype("float32")

    sensors = hourly["location_id"].to_numpy()
    y = hourly["pedestriancount"].to_numpy(dtype=float)
    ts = hourly["ts_utc"]

    n_folds = folds if folds is not None else tcfg["folds"]
    plan = fold_plan(sensors, n_folds, 600_000, tcfg["seed"])

    params = _model_params(cfg, variant)
    if n_iter is not None:
        params = dict(params)
        params["max_iter"] = n_iter

    window = calibration_window(sensors, ts, cal_days)

    # OOF predictions: Poisson (mean) + quantile models for 80% intervals
    oof_mean = np.full(len(hourly), np.nan)
    oof_q10 = np.full(len(hourly), np.nan)
    oof_q90 = np.full(len(hourly), np.nan)

    for train_pos, test_pos in plan:
        def _build_x(pos: np.ndarray) -> np.ndarray:
            return np.nan_to_num(
                design_matrix(hourly, pos, osm, station_cap_m, all_row_cols), nan=0.0
            )

        x_train, x_test = _build_x(train_pos), _build_x(test_pos)
        y_train = y[train_pos]

        base_params = {k: v for k, v in params.items() if k not in ("early_stopping",)}

        m_mean = HistGradientBoostingRegressor(loss="poisson", **base_params)
        m_mean.fit(x_train, y_train)
        oof_mean[test_pos] = np.maximum(m_mean.predict(x_test), 0.0)

        for q, arr in [(0.1, oof_q10), (0.9, oof_q90)]:
            m_q = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, **{k: v for k, v in base_params.items() if k != "loss"}
            )
            m_q.fit(x_train, y_train)
            arr[test_pos] = np.maximum(m_q.predict(x_test), 0.0)

    # Score on non-calibration-window rows
    scored = ~window & ~np.isnan(oof_mean)
    w = wape(y[scored], oof_mean[scored])
    ml = mae_log(y[scored], oof_mean[scored])
    cov = interval_coverage(y[scored], oof_q10[scored], oof_q90[scored])

    # Save OOF predictions
    effective_out = out_dir or _out_dir(variant)
    effective_out.mkdir(parents=True, exist_ok=True)
    oof_df = pd.DataFrame(
        {
            "location_id": sensors,
            "ts_utc": ts.to_numpy(),
            "pedestriancount": y,
            "oof_mean": oof_mean,
            "oof_q10": oof_q10,
            "oof_q90": oof_q90,
            "in_calibration_window": window,
        }
    )
    oof_df.to_parquet(effective_out / "oof_predictions.parquet", index=False)

    # Train final model on all rows
    x_all = np.nan_to_num(
        design_matrix(hourly, np.arange(len(hourly)), osm, station_cap_m, all_row_cols),
        nan=0.0,
    )
    final_model = HistGradientBoostingRegressor(
        loss="poisson", **{k: v for k, v in params.items() if k not in ("early_stopping",)}
    )
    final_model.fit(x_all, y)
    joblib.dump(final_model, effective_out / "model.pkl")
    (effective_out / "feature_names.json").write_text(json.dumps(feat_names))

    return {
        "wape": w,
        "mae_log": ml,
        "coverage": cov,
        "out_dir": str(effective_out),
        "n_rows_scored": int(scored.sum()),
        "params": params,
        "feature_names": feat_names,
        "final_model": final_model,
        "oof_mean": oof_mean,
        "oof_q10": oof_q10,
        "oof_q90": oof_q90,
        "y": y,
        "scored": scored,
        "hourly": hourly,
        "osm": osm,
        "all_row_cols": all_row_cols,
        "station_cap_m": station_cap_m,
    }


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=[*VARIANTS, "both"], default="both")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--if-older-than", type=float, metavar="DAYS")
    args = ap.parse_args()

    import mlflow

    cfg, _, _ = _load_configs()

    if args.if_older_than is not None:
        # Check the meta file for all requested variants
        variants_to_run = []
        for v in (list(VARIANTS) if args.variant == "both" else [args.variant]):
            meta_path = _out_dir(v) / "train_meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                age = datetime.now(UTC) - datetime.fromisoformat(meta.get("run_at", "2000-01-01"))
                if age < timedelta(days=args.if_older_than):
                    continue
            variants_to_run.append(v)
        if not variants_to_run:
            print(f"Models are under {args.if_older_than} days old; skipped.")
            return
    else:
        variants_to_run = list(VARIANTS) if args.variant == "both" else [args.variant]

    if args.tune:
        import mlflow as _mlf

        from src.models.flow import tune
        _mlf.set_tracking_uri(f"file:{REPO_ROOT / 'mlruns'}")
        for v in variants_to_run:
            fr = load(variant=v)
            best = tune.build_study(fr.hourly, fr.osm, cfg, v)
            if "best_params" not in cfg:
                cfg["best_params"] = {}
            cfg["best_params"][v] = best
        MODEL_YAML.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    mlflow.set_tracking_uri(f"file:{REPO_ROOT / 'mlruns'}")
    mlflow.set_experiment("stoa_flow_model")
    bias_cfg = yaml.safe_load(BIAS_YAML.read_text())

    for variant in variants_to_run:
        frame = load(variant=variant)
        hourly, osm = frame.hourly, frame.osm

        t = time.perf_counter()
        with mlflow.start_run(run_name=f"flow_{variant}") as run:
            results = run_variant(hourly, osm, cfg, variant)
            mlflow.log_params(
                {f"param_{k}": v for k, v in results["params"].items()}
            )
            mlflow.log_metrics(
                {
                    "wape": results["wape"],
                    "mae_log": results["mae_log"],
                    "interval_coverage": results["coverage"],
                }
            )

            # Baseline ladder
            baseline_preds = run_baselines(hourly, osm, cfg)
            all_preds = {
                "betweenness": baseline_preds["betweenness"],
                "space_syntax": baseline_preds["space_syntax"],
                f"learned_{variant}": results["oof_mean"],
            }
            ladder = baseline_ladder(results["y"], all_preds, results["scored"])
            ladder_csv = ladder.to_csv(index=False)
            mlflow.log_text(ladder_csv, "baseline_ladder.csv")
            print(f"\nBaseline ladder ({variant}):\n{ladder.to_string(index=False)}\n")

            # SHAP on 1000-row sample
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(
                np.flatnonzero(results["scored"]),
                size=min(1000, int(results["scored"].sum())),
                replace=False,
            )
            x_sample = np.nan_to_num(
                design_matrix(
                    results["hourly"],
                    sample_idx,
                    results["osm"],
                    results["station_cap_m"],
                    results["all_row_cols"],
                ),
                nan=0.0,
            )
            shap_summary(results["final_model"], x_sample, results["feature_names"])

            # Calibration curve
            cal_df = calibration_curve(
                hourly,
                results["oof_mean"],
                cfg["calibration_curve"]["k_values"],
                cfg["calibration_curve"]["reps"],
                cfg["calibration_curve"]["seed"],
            )
            cal_csv = cal_df.to_csv(index=False)
            mlflow.log_text(cal_csv, "calibration_curve.csv")

            # Transfer precinct eval
            tp_df = transfer_precinct_eval(
                hourly=results["hourly"],
                osm=results["osm"],
                model=results["final_model"],
                holdout_precincts=bias_cfg["holdout_precincts"],
                row_cols=results["all_row_cols"],
                station_cap_m=results["station_cap_m"],
                metro_tunnel_date=cfg["transfer"]["metro_tunnel_date"],
            )
            if not tp_df.empty:
                mlflow.log_text(tp_df.to_csv(index=False), "transfer_precincts.csv")
                print(f"Transfer precincts:\n{tp_df.to_string(index=False)}\n")

            # Gate check (informational here; enforced in registry.py)
            ss_mae = baseline_ladder(
                results["y"],
                {"space_syntax": baseline_preds["space_syntax"]},
                results["scored"],
            ).iloc[0]["mae_log"]
            passed, msg = check_gates(
                results["mae_log"], float(ss_mae), results["coverage"], cfg
            )
            mlflow.log_metric("gates_passed", int(passed))
            mlflow.set_tag("gates_message", msg)
            mlflow.log_param("run_id", run.info.run_id)

            mlflow.sklearn.log_model(results["final_model"], "model")
            runtime = time.perf_counter() - t

        # Write local meta
        out_dir = Path(results["out_dir"])
        (out_dir / "train_meta.json").write_text(
            json.dumps(
                {
                    "variant": variant,
                    "run_at": datetime.now(UTC).isoformat(),
                    "mlflow_run_id": run.info.run_id,
                    "wape": results["wape"],
                    "mae_log": results["mae_log"],
                    "coverage": results["coverage"],
                    "runtime_s": round(runtime, 1),
                    "params": results["params"],
                    "gates_passed": passed,
                    "gates_message": msg,
                },
                indent=2,
            )
        )
        print(
            f"{variant}: WAPE={results['wape']:.4f} MAE-log={results['mae_log']:.4f} "
            f"coverage={results['coverage']:.3f} ({runtime:.0f} s)  {msg}"
        )


if __name__ == "__main__":
    main()
