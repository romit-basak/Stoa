"""A Markdown training report for every flow model, written each time the model is trained.

Each report covers the data the model saw (its period, and how many sensors it never saw),
what it minimises and how far that went from the first tree to the last, its learned
parameters and its hyperparameters. src.models.flow.train writes one per candidate and
baseline, and src.models.flow.select one for the selected model; each is saved under
<output_dir>/<variant>/reports/ and logged to the model's MLflow run as `model_report.md`.

Loss and cost, as the reports use the words: the loss is the per-row function the training
minimises (Poisson deviance for the mean models, pinball loss for the 10% and 90% interval
models, squared error on log counts for the baselines); the cost is its mean over a set of
rows. The boosted models add an L2 penalty on leaf values (`reg_lambda`, `l2_regularization`)
during training, which the reported costs leave out so they stay comparable across families.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Share of the trees at which the cost is recorded, besides "before the first tree".
CHECKPOINTS = (0.01, 0.1, 0.25, 0.5, 1.0)
SAMPLE_ROWS = 20_000


# --------------------------------------------------------------------------------------------
# Loss functions
# --------------------------------------------------------------------------------------------


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Mean Poisson deviance, 2 * (y log(y / mu) - (y - mu)), with mu kept above zero."""
    y, mu = np.asarray(y, float), np.maximum(np.asarray(mu, float), 1e-6)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    return float(np.mean(2 * (term - (y - mu))))


def pinball(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    diff = np.asarray(y, float) - np.asarray(q, float)
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def loss_for(
    quantile: float | None,
) -> tuple[str, Callable[[np.ndarray, np.ndarray], float]]:
    if quantile:
        return f"pinball loss at {quantile:g}", lambda y, p: pinball(y, p, quantile)
    return "Poisson deviance", poisson_deviance


def sample(rows: np.ndarray, n: int = SAMPLE_ROWS, seed: int = 0) -> np.ndarray:
    if len(rows) <= n:
        return rows
    return np.sort(np.random.default_rng(seed).choice(rows, n, replace=False))


# --------------------------------------------------------------------------------------------
# Boosted models: the cost tree by tree, and what they learned
# --------------------------------------------------------------------------------------------


KINDS = {
    "HistGradientBoostingRegressor": "hgb",
    "LGBMRegressor": "lgbm",
    "XGBRegressor": "xgb",
    "RandomForestRegressor": "rf",
}


def kind(family: str, model: Any) -> str:
    """The model's own type, which can differ from its family: rf's interval models are
    HistGradientBoosting."""
    return KINDS.get(type(model).__name__, family)


def n_trees(family: str, model: Any) -> int:
    family = kind(family, model)
    if family == "rf":
        return len(model.estimators_)
    if family == "hgb":
        return int(model.n_iter_)
    if family == "lgbm":
        return int(model.booster_.current_iteration())
    if family == "xgb":
        return int(model.get_booster().num_boosted_rounds())
    raise ValueError(family)


def staged(
    family: str, model: Any, x: np.ndarray, at: Sequence[int]
) -> dict[int, np.ndarray]:
    """Predictions using only the first k trees, for each k in `at` (for a forest, the mean of
    its first k trees)."""
    want = set(at)
    family = kind(family, model)
    if family == "rf":
        total = np.cumsum([t.predict(x) for t in model.estimators_], axis=0)
        return {k: total[k - 1] / k for k in sorted(want)}
    if family == "hgb":
        return {
            k: p for k, p in enumerate(model.staged_predict(x), start=1) if k in want
        }
    if family == "lgbm":
        return {k: model.predict(x, num_iteration=k) for k in sorted(want)}
    if family == "xgb":
        return {k: model.predict(x, iteration_range=(0, k)) for k in sorted(want)}
    raise ValueError(family)


def cost_curve(
    family: str,
    model: Any,
    sets: Mapping[str, tuple[np.ndarray, np.ndarray]],
    start: float,
    quantile: float | None = None,
) -> pd.DataFrame:
    """The cost on each named (x, y) set before the first tree and after k trees.

    "Before the first tree" predicts `start` for every row: the training mean for a mean
    model, the training quantile for an interval model, as the boosters start from.
    """
    _, loss = loss_for(quantile)
    total = n_trees(family, model)
    at = sorted({max(1, round(c * total)) for c in CHECKPOINTS})
    rows = [
        {
            "trees": 0,
            **{n: loss(y, np.full(len(y), start)) for n, (_, y) in sets.items()},
        }
    ]
    preds = {n: staged(family, model, x, at) for n, (x, _) in sets.items()}
    for k in at:
        rows.append(
            {"trees": k, **{n: loss(y, preds[n][k]) for n, (_, y) in sets.items()}}
        )
    return pd.DataFrame(rows)


def learned(family: str, model: Any) -> dict[str, Any]:
    """Counts of what the fitted trees hold: trees, leaves (one learned value each) and splits."""
    family = kind(family, model)
    out: dict[str, Any] = {
        "trees": n_trees(family, model),
        "features_in": int(model.n_features_in_),
    }
    try:
        if family == "hgb":
            nodes = [p.nodes for (p,) in model._predictors]
            leaves = sum(int(n["is_leaf"].sum()) for n in nodes)
            used = {
                int(f)
                for n in nodes
                for f in n["feature_idx"][~n["is_leaf"].astype(bool)]
            }
        elif family == "lgbm":
            trees = model.booster_.dump_model()["tree_info"]
            leaves = sum(int(t["num_leaves"]) for t in trees)
            used = set(np.flatnonzero(model.booster_.feature_importance("split")))
        elif family == "rf":
            trees = [t.tree_ for t in model.estimators_]
            leaves = sum(int(t.n_leaves) for t in trees)
            used = {int(f) for t in trees for f in t.feature if f >= 0}
        else:
            df = model.get_booster().trees_to_dataframe()
            leaves = int((df["Feature"] == "Leaf").sum())
            used = set(df.loc[df["Feature"] != "Leaf", "Feature"])
        out.update(
            leaves=leaves,
            splits=leaves - out["trees"],
            learned_values=2 * leaves - out["trees"],
            features_used=len(used),
        )
    except (AttributeError, KeyError, TypeError):
        pass  # the counts are a summary; a library that hides its trees just leaves them out
    return out


def baseline_curve(
    model: Any,
    hourly: pd.DataFrame,
    osm: pd.DataFrame,
    y: np.ndarray,
    sets: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """A baseline's cost (squared error on log1p counts), fitted in one step: "null" predicts
    every row the training mean log count, "fitted" is the rung."""
    ly_mean = float(np.mean(np.log1p(y[next(iter(sets.values()))])))
    out = []
    for stage in ("null (training mean)", "fitted"):
        row: dict[str, Any] = {"fit": stage}
        for n, rows in sets.items():
            ly = np.log1p(y[rows])
            p = (
                np.full(len(rows), ly_mean)
                if stage.startswith("null")
                else np.log1p(model.predict(hourly, osm, rows))
            )
            row[n] = float(np.mean((ly - p) ** 2))
        out.append(row)
    return pd.DataFrame(out)


def baseline_learned(model: Any) -> dict[str, Any]:
    ridge = model.ridge_
    return {
        "hour_of_week_profile_values": len(model.profile_),
        "ridge_coefficients": int(np.size(ridge.coef_)),
        "ridge_intercept": float(ridge.intercept_),
        "largest_coefficients": ", ".join(
            f"{c} {w:+.3f}"
            for w, c in sorted(
                zip(np.ravel(ridge.coef_), model.columns), key=lambda t: -abs(t[0])
            )[:5]
        ),
    }


def mean_curve(curves: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Fold curves averaged position by position (folds can differ by a tree when rounding)."""
    n = min(len(c) for c in curves)
    stack = [c.iloc[:n].reset_index(drop=True) for c in curves]
    stage = stack[0].columns[0]
    nums = [s.drop(columns=stage) for s in stack]
    out = sum(nums[1:], nums[0]) / len(nums)
    first = stack[0][stage]
    if stage == "trees":
        first = (sum(s[stage] for s in stack) / len(stack)).round().astype(int)
    return pd.concat([first, out], axis=1)


def mean_counts(dicts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-fold parameter counts averaged; text values taken from the first fold."""
    out: dict[str, Any] = {}
    for k, v in dicts[0].items():
        vals = [d[k] for d in dicts if k in d]
        out[k] = (
            float(np.mean(vals))
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            else v
        )
        if isinstance(v, int) and not isinstance(v, bool):
            out[k] = round(out[k])
    return out


# --------------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------------


@dataclass
class DataFacts:
    """What the model saw: the period, its rows, and which sensors it never saw."""

    period: tuple[str, str]
    train_rows: int
    sensors_total: int
    sensors_trained: int
    hidden: dict[str, list[int]] = field(default_factory=dict)
    note: str = ""
    rows_label: str = "sampled"


def data_facts(
    data: Any,
    rows: np.ndarray,
    hidden: dict[str, np.ndarray],
    note: str = "",
    train_rows: int | None = None,
    rows_label: str = "sampled",
) -> DataFacts:
    """Facts for training rows `rows` of `data` (src.models.flow.data.Data); `hidden` names
    each group of sensors the model never saw. `train_rows` overrides the row count shown
    (fold models: one fold's sample, while `rows` is every fold's)."""
    local = pd.to_datetime(data.ts.to_numpy()[rows], utc=True).tz_convert(
        "Australia/Melbourne"
    )
    return DataFacts(
        period=(local.min().strftime("%Y-%m-%d"), local.max().strftime("%Y-%m-%d")),
        train_rows=len(rows) if train_rows is None else train_rows,
        sensors_total=len(np.unique(data.sensors)),
        sensors_trained=len(np.unique(data.sensors[rows])),
        hidden={k: sorted(int(s) for s in np.unique(v)) for k, v in hidden.items()},
        note=note,
        rows_label=rows_label,
    )


def fold_hidden(data: Any) -> dict[str, np.ndarray]:
    """For fold models: each fold's test sensors, and the holdout precincts no fold sees."""
    out = {
        f"fold {k} (scored on them)": data.sensors[data.fold == k]
        for k in sorted(set(data.fold) - {-1})
    }
    out["holdout precincts and neighbours (no fold)"] = data.sensors[data.fold < 0]
    return out


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    rule = "|" + "---|" * len(df.columns)
    body = [
        "| " + " | ".join(_fmt(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([head, rule, *body])


def render(
    name: str,
    what: str,
    facts: DataFacts,
    loss_name: str,
    cost_note: str,
    curves: Mapping[str, pd.DataFrame],
    parameters: Mapping[str, Any],
    hyperparameters: Mapping[str, Any],
    tuning: str,
    metrics: Mapping[str, float] | None = None,
    created_at: str = "",
) -> str:
    """The report as Markdown. `curves` maps a heading to a cost table whose first column is
    the stage (`trees`, 0 = before the first tree; or a label for models fitted in one step)
    and whose other columns are row sets. Start is the first row, end the last."""
    lines = [f"# {name}", "", what, ""]
    if created_at:
        lines += [f"Trained {created_at}.", ""]
    lines += [
        "## Data",
        "",
        f"- Period: {facts.period[0]} to {facts.period[1]} (local dates of the training rows)",
        f"- Training rows: {facts.train_rows:,} ({facts.rows_label})",
        f"- Sensors: {facts.sensors_trained} trained on, of {facts.sensors_total} in the frame",
    ]
    for group, ids in facts.hidden.items():
        shown = ", ".join(map(str, ids[:40])) + (" ..." if len(ids) > 40 else "")
        lines.append(f"- Never seen, {group}: {len(ids)} sensors ({shown})")
    if facts.note:
        lines.append(f"- {facts.note}")
    lines += [
        "",
        "## Loss and cost",
        "",
        f"- Loss function (per row): {loss_name}",
        f"- Cost function: {cost_note}",
        "",
    ]
    for heading, df in curves.items():
        stage = df.columns[0]
        cols = [c for c in df.columns if c != stage]
        first, last = df.iloc[0], df.iloc[-1]
        lines.append(f"**{heading}**")
        lines.append("")
        lines.append("| Rows | Start | End | Change |")
        lines.append("|---|---|---|---|")
        for c in cols:
            change = (last[c] - first[c]) / first[c] if first[c] else float("nan")
            lines.append(
                f"| {c} | {_fmt(float(first[c]))} | {_fmt(float(last[c]))} | {change:+.1%} |"
            )
        if stage == "trees":
            lines += ["", "By trees (0 is before the first tree):", "", _table(df), ""]
        else:
            lines.append("")
    lines += ["## Learned parameters", ""]
    lines += [f"- {k.replace('_', ' ')}: {_fmt(v)}" for k, v in parameters.items()]
    lines += ["", "## Hyperparameters", "", tuning, ""]
    lines += [f"- `{k}`: {_fmt(v)}" for k, v in sorted(hyperparameters.items())]
    if metrics:
        keep = [
            k
            for k in (
                "log_mae_cv",
                "wape_cv",
                "wape_cv_cal",
                "coverage80_cv",
                "log_mae_holdout",
                "wape_holdout",
            )
            if k in metrics
        ]
        if keep:
            lines += ["", "## Scores", ""]
            lines += [f"- `{k}`: {_fmt(float(metrics[k]))}" for k in keep]
    return "\n".join(lines) + "\n"


def tuning_text(tuned: bool, space: Mapping[str, Any] | None) -> str:
    if not tuned:
        return "Defaults from `configs/models.yaml` (not tuned in this run)."
    parts = []
    for k, s in (space or {}).items():
        if s["type"] == "choice":
            parts.append(f"`{k}` in {s['choices']}")
        else:
            log = ", log scale" if s["type"] == "float_log" else ""
            parts.append(f"`{k}` {s['low']} to {s['high']}{log}")
    return (
        "Tuned with Optuna (TPE) on inner folds of the fold sensors; best values below. "
        "Search space: " + "; ".join(parts) + "."
    )
