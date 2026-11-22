"""Load a registered flow model (the bundle from src/models/flow/bundle.py) for serving.

`uri` is anything mlflow.pyfunc.load_model takes: `models:/stoa-flow-rich@champion` through
a tunnel to the team's MLflow during development, or a local directory holding a saved
bundle (what the serving image will carry). Loading checks that the model expects exactly
the columns src/api/features.py builds for its variant, so a schema change in training
can't reach serving unnoticed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.features import training_frame as tf


@dataclass
class LoadedModel:
    variant: str
    uri: str
    columns: list[str]
    predictor: Any  # has .predict(DataFrame) -> DataFrame with predicted, lower, upper
    name: str | None = None
    version: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def label(self) -> str:
        if self.name and self.version:
            return f"{self.name} v{self.version}"
        return self.uri

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.predictor.predict(features[self.columns])


def expected_columns(variant: str) -> list[str]:
    return (
        tf.row_features(variant)
        + tf.derived_features(variant)
        + tf.segment_features(variant)
    )


def registry_info(uri: str) -> tuple[str | None, str | None, dict[str, float]]:
    """(name, version, the training run's metrics) for a models:/ URI, else Nones."""
    m = re.fullmatch(r"models:/([^/@]+)(?:@([^/]+)|/(\d+))", uri)
    if not m:
        return None, None, {}
    from mlflow import MlflowClient

    client = MlflowClient()
    name, alias, number = m.groups()
    mv = (
        client.get_model_version_by_alias(name, alias)
        if alias
        else client.get_model_version(name, number)
    )
    metrics = client.get_run(mv.run_id).data.metrics if mv.run_id else {}
    keep = ("log_mae_cv", "coverage80_cv", "log_mae_holdout", "wape_cv")
    return name, mv.version, {k: v for k, v in metrics.items() if k in keep}


def load(variant: str, uri: str) -> LoadedModel:
    import mlflow.pyfunc

    pyfunc = mlflow.pyfunc.load_model(uri)
    bundle = pyfunc.unwrap_python_model()
    columns = list(getattr(bundle, "columns", []))
    want = expected_columns(variant)
    if set(columns) != set(want):
        missing = sorted(set(want) - set(columns))[:5]
        extra = sorted(set(columns) - set(want))[:5]
        raise ValueError(
            f"{uri} doesn't expect the {variant} features: missing {missing}, extra {extra}"
        )
    name, version, metrics = registry_info(uri)
    return LoadedModel(
        variant, uri, columns, bundle_predictor(bundle), name, version, metrics
    )


def bundle_predictor(bundle: Any) -> Any:
    class _Predictor:
        def predict(self, features: pd.DataFrame) -> pd.DataFrame:
            return bundle.predict(None, features)

    return _Predictor()
