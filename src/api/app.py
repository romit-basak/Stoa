"""The Stoa prediction API.

    uvicorn src.api.app:app --port 8000     # docs at http://localhost:8000/docs

Endpoints:
- GET  /health   200 once at least one model is loaded, 503 before;
- GET  /model    which model version serves each variant, and its training metrics;
- POST /predict  predicted pedestrians per segment and hour, with an 80% range.

Settings (environment):
- STOA_MODEL_URI_RICH, STOA_MODEL_URI_PORTABLE: where to load each variant from (default the
  registry's `champion`, e.g. models:/stoa-flow-rich@champion; needs MLFLOW_TRACKING_URI,
  and the tunnel to the VM during development). A variant set to "" isn't loaded.
- STOA_SITE: the site served (default the training site, counts_hourly.site).

Predictions are planning guidance (CLAUDE.md core rule 7): a forecast of passersby with a
range, not a count anyone measured.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from src.api import model as model_mod
from src.api.features import BadHours, SiteTables, UnknownSegments, build_features
from src.api.weather import ForecastWeather, WeatherUnavailable

log = logging.getLogger("stoa.api")

MAX_SEGMENTS = 500
MAX_HOURS = 168
MAX_ROWS = 20_000
VARIANTS = ("rich", "portable")


class PredictRequest(BaseModel):
    site: str | None = Field(
        None, description="The site served (GET /model); default that site"
    )
    variant: Literal["rich", "portable"] = Field(
        "rich",
        description="rich: Melbourne's own layers; portable: features any city has",
    )
    segment_ids: list[str] = Field(
        ..., min_length=1, max_length=MAX_SEGMENTS, examples=[["4187260-0"]]
    )
    hours: list[datetime] = Field(
        ...,
        min_length=1,
        max_length=MAX_HOURS,
        description="Local clock times at the site, without a UTC offset",
        examples=[["2026-11-24T08:00"]],
    )

    @model_validator(mode="after")
    def _size(self) -> PredictRequest:
        if len(self.segment_ids) * len(self.hours) > MAX_ROWS:
            raise ValueError(f"segments x hours must be at most {MAX_ROWS}")
        return self


class Prediction(BaseModel):
    segment_id: str
    hour: str
    predicted_count: float
    lower: float
    upper: float
    model_version: str


class PredictResponse(BaseModel):
    site: str
    variant: str
    model_version: str
    predictions: list[Prediction]
    sources: dict[str, Any]
    note: str = (
        "Planning guidance: predicted passersby per hour with an 80% range, "
        "not a measured count."
    )


def load_models() -> dict[str, model_mod.LoadedModel | str]:
    """Each variant's model, or the reason it didn't load."""
    out: dict[str, model_mod.LoadedModel | str] = {}
    for v in VARIANTS:
        uri = os.environ.get(
            f"STOA_MODEL_URI_{v.upper()}", f"models:/stoa-flow-{v}@champion"
        )
        if not uri:
            continue
        try:
            out[v] = model_mod.load(v, uri)
            log.info("loaded %s from %s", out[v].label, uri)
        except Exception as e:  # noqa: BLE001 (serve the other variant rather than none)
            out[v] = f"{type(e).__name__}: {e}"
            log.error("couldn't load %s from %s: %s", v, uri, e)
    return out


def create_app(
    tables: SiteTables | None = None,
    models: dict[str, model_mod.LoadedModel | str] | None = None,
    weather: ForecastWeather | None = None,
) -> FastAPI:
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        site = os.environ.get("STOA_SITE") or None
        state["tables"] = tables or SiteTables.load(site)
        state["weather"] = weather or ForecastWeather(
            state["tables"].data_cfg,
            state["tables"].site,
            interval_vars=state["tables"].fcfg["counts_hourly"][
                "weather_interval_vars"
            ],
        )
        state["models"] = models if models is not None else load_models()
        yield

    app = FastAPI(
        title="Stoa prediction API",
        summary="Hourly pedestrian flow per path segment, from the registered flow model.",
        lifespan=lifespan,
    )

    def ready() -> dict[str, model_mod.LoadedModel]:
        return {
            k: m
            for k, m in state.get("models", {}).items()
            if isinstance(m, model_mod.LoadedModel)
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        loaded = ready()
        if not loaded:
            raise HTTPException(503, "no model loaded")
        return {"status": "ok", "variants": sorted(loaded)}

    @app.get("/model")
    def model_info() -> dict[str, Any]:
        out: dict[str, Any] = {}
        for v, m in state.get("models", {}).items():
            if isinstance(m, model_mod.LoadedModel):
                out[v] = {
                    "name": m.name,
                    "version": m.version,
                    "uri": m.uri,
                    "features": len(m.columns),
                    "metrics": m.metrics,
                }
            else:
                out[v] = {"error": m}
        return {"site": state["tables"].site, "models": out}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        m = ready().get(req.variant)
        if m is None:
            raise HTTPException(503, f"the {req.variant} model isn't loaded")
        tables: SiteTables = state["tables"]
        if req.site not in (None, tables.site):
            raise HTTPException(422, f"this API serves {tables.site}, not {req.site}")
        hours = [pd.Timestamp(h) for h in req.hours]
        try:
            from src.api.features import localize

            times = localize(hours, tables.timezone)
            wx = state["weather"].for_hours(times["ts_utc"])
            rows, x, sources = build_features(
                tables, req.segment_ids, hours, wx, req.variant
            )
        except UnknownSegments as e:
            raise HTTPException(404, {"unknown_segment_ids": e.segments[:50]}) from e
        except BadHours as e:
            raise HTTPException(422, str(e)) from e
        except WeatherUnavailable as e:
            raise HTTPException(503, str(e)) from e
        out = m.predict(x)
        hour = rows["ts_local"].dt.strftime("%Y-%m-%dT%H:%M")
        preds = [
            Prediction(
                segment_id=s,
                hour=h,
                predicted_count=round(float(p), 1),
                lower=round(float(lo), 1),
                upper=round(float(hi), 1),
                model_version=m.label,
            )
            for s, h, p, lo, hi in zip(
                rows["segment_id"], hour, out["predicted"], out["lower"], out["upper"]
            )
        ]
        return PredictResponse(
            site=tables.site,
            variant=req.variant,
            model_version=m.label,
            predictions=preds,
            sources=sources,
        )

    return app


app = create_app()
