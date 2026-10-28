"""FastAPI service: score an applicant on either track, with reason codes, and log it.

    POST /score              {"track": "lending_club" | "mobile_money", "applicant": {...}}
    GET  /model              what each served model is, its cutoff and headline metrics
    GET  /examples/{track}   sample applicants (what the dashboard's what-if starts from)
    GET  /decisions          the most recent logged decisions
    GET  /health

Run locally with `uv run uvicorn crisk.serve.app:app --reload` after `crisk export` (and
`crisk migrate` for decision logging). No authentication: an API-key header would come
before any public deployment. The mobile-money track is trained on SYNTHETIC data and
every response from it says so.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import Engine, insert, select, text

from crisk.config import get_settings
from crisk.db.models import Decision
from crisk.serve.predictor import TrackModel, load_models

log = logging.getLogger("crisk.api")
Track = Literal["lending_club", "mobile_money"]
Value = float | int | str | None


class ScoreRequest(BaseModel):
    track: Track
    applicant: dict[str, Value] = Field(
        description="Model features by name; any feature left out is treated as missing."
    )
    applicant_ref: str | None = Field(None, max_length=64)


class Reason(BaseModel):
    code: str
    text: str
    actionable: bool
    weight: float = Field(description="Log-odds the reason added to the PD (SHAP)")


class ScoreResponse(BaseModel):
    track: Track
    model_version: str
    synthetic: bool
    pd: float = Field(description="Calibrated probability of default")
    score: int = Field(description="Points on the 600 = 50:1 odds, 20 points to double scale")
    risk_band: str
    decision: Literal["approve", "decline"]
    cutoff: float
    reasons: list[Reason] = Field(description="Up to four adverse-action reasons (declines)")
    missing_features: list[str]
    decision_id: int | None = Field(None, description="Row in the decisions table, if logged")


def create_app(models: dict[str, TrackModel] | None = None, engine: Engine | None = None):
    """App factory. Tests inject models and an engine; production loads both."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = get_settings()
        app.state.models = models if models is not None else load_models(Path(settings.serving_dir))
        app.state.log_decisions = settings.log_decisions
        if engine is not None:
            app.state.engine = engine
        else:
            from crisk.db.session import get_engine

            app.state.engine = get_engine()
        yield
        if engine is None:
            app.state.engine.dispose()

    app = FastAPI(
        title="crisk",
        summary="Credit-risk scoring: Lending Club PD and a synthetic thin-file track",
        version="0.8.0",
        lifespan=lifespan,
    )
    _routes(app)
    return app


def _model(request: Request, track: str) -> TrackModel:
    model = request.app.state.models.get(track)
    if model is None:
        raise HTTPException(503, f"no model exported for track {track!r} (run crisk export)")
    return model


def _log(request: Request, model: TrackModel, req: ScoreRequest, scored) -> int | None:
    if not request.app.state.log_decisions:
        return None
    row = {
        "track": model.track,
        "model_version": model.meta["version"],
        "synthetic": bool(model.meta["synthetic"] or model.meta.get("demo")),
        "applicant_ref": req.applicant_ref,
        "pd": scored.pd,
        "score": scored.score,
        "risk_band": scored.risk_band,
        "decision": scored.decision,
        "cutoff": model.meta["cutoff"],
        "reasons": scored.reasons,
        "applicant": req.applicant,
    }
    try:
        with request.app.state.engine.begin() as conn:
            return conn.execute(insert(Decision).values(**row).returning(Decision.id)).scalar_one()
    except Exception as exc:  # scoring must not fail because the audit log is down
        log.warning("decision not logged: %s", exc)
        return None


def _routes(app: FastAPI) -> None:
    @app.get("/health")
    def health(request: Request) -> dict:
        try:
            with request.app.state.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db = True
        except Exception:
            db = False
        tracks = sorted(request.app.state.models)
        return {"status": "ok" if tracks else "no models", "tracks": tracks, "database": db}

    @app.get("/model")
    def model_info(request: Request) -> dict:
        keep = (
            "track",
            "version",
            "exported_at",
            "model_name",
            "trained_on",
            "synthetic",
            "demo",
            "cutoff",
            "cutoff_rule",
            "currency",
            "metrics",
            "risk_bands",
            "score_scale",
            "features",
        )
        return {
            t: {k: m.meta.get(k) for k in keep}
            | {
                "reasons": [
                    {k: r[k] for k in ("code", "text", "actionable")} for r in m.meta["reasons"]
                ]
            }
            for t, m in request.app.state.models.items()
        }

    @app.get("/examples/{track}")
    def examples(request: Request, track: Track) -> list[dict]:
        return _model(request, track).examples

    @app.post("/score")
    def score(request: Request, req: ScoreRequest) -> ScoreResponse:
        model = _model(request, req.track)
        unknown = sorted(set(req.applicant) - set(model.features) - {"annual_inc", "loan_amnt"})
        if unknown:
            raise HTTPException(422, f"unknown features for {req.track}: {unknown}")
        scored = model.score([req.applicant])[0]
        decision_id = _log(request, model, req, scored)
        return ScoreResponse(
            track=req.track,
            model_version=model.meta["version"],
            synthetic=bool(model.meta["synthetic"] or model.meta.get("demo")),
            pd=scored.pd,
            score=scored.score,
            risk_band=scored.risk_band,
            decision=scored.decision,
            cutoff=model.meta["cutoff"],
            reasons=scored.reasons,
            missing_features=scored.missing_features,
            decision_id=decision_id,
        )

    @app.get("/decisions")
    def decisions(request: Request, limit: int = Query(20, ge=1, le=200)) -> list[dict]:
        cols = [
            Decision.id,
            Decision.created_at,
            Decision.track,
            Decision.applicant_ref,
            Decision.pd,
            Decision.score,
            Decision.risk_band,
            Decision.decision,
        ]
        try:
            with request.app.state.engine.connect() as conn:
                rows = conn.execute(select(*cols).order_by(Decision.id.desc()).limit(limit))
                return [dict(r._mapping) for r in rows]
        except Exception as exc:
            raise HTTPException(503, f"decision log unavailable: {exc}") from exc


app = create_app()
