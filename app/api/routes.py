from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.requests import AgentQueryRequest, AlertRequest, CompareRequest, ForecastRequest, TrainPipelineRequest
from app.schemas.responses import (
    AgentResponse,
    AlertResponse,
    ComparisonResponse,
    EntityContextResponse,
    ForecastResponse,
    HistoryResponse,
    TrainPipelineResponse,
)
from app.schemas.common import EntityType, TargetMetric
from app.services.entity_context import get_entity_context
from app.services.forecasting import ForecastService
from app.services.pipeline import TrainingPipelineService
from app.services.router import AgentRouterService


router = APIRouter()


@router.post("/pipeline/train", response_model=TrainPipelineResponse)
def train_pipeline(request: TrainPipelineRequest, db: Session = Depends(get_db)) -> TrainPipelineResponse:
    try:
        return TrainingPipelineService(db).run(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/forecasts/entity", response_model=ForecastResponse)
def forecast_entity(request: ForecastRequest, db: Session = Depends(get_db)) -> ForecastResponse:
    try:
        service = ForecastService(db)
        return service.forecast_entity(
            entity_type=request.entity_type,
            entity_id=request.entity_id.upper(),
            horizon_weeks=request.horizon_weeks,
            target_metric=request.target_metric,
            price_multiplier=request.price_multiplier,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/forecasts/compare", response_model=ComparisonResponse)
def compare_entities(request: CompareRequest, db: Session = Depends(get_db)) -> ComparisonResponse:
    try:
        service = ForecastService(db)
        return service.compare_entities(
            entity_type=request.entity_type,
            entity_ids=[entity_id.upper() for entity_id in request.entity_ids],
            horizon_weeks=request.horizon_weeks,
            target_metric=request.target_metric,
            price_multiplier=request.price_multiplier,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/history/entity", response_model=HistoryResponse)
def entity_history(
    entity_type: EntityType,
    entity_id: str,
    target_metric: TargetMetric = TargetMetric.REVENUE,
    lookback_weeks: int = Query(default=4, ge=4, le=52),
    db: Session = Depends(get_db),
) -> HistoryResponse:
    try:
        service = ForecastService(db)
        return service.entity_history(
            entity_type=entity_type,
            entity_id=entity_id.upper(),
            target_metric=target_metric,
            lookback_weeks=lookback_weeks,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/context/entity", response_model=EntityContextResponse)
def entity_context(entity_type: EntityType, entity_id: str) -> EntityContextResponse:
    try:
        return get_entity_context(entity_type=entity_type, entity_id=entity_id.upper())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/alerts/declining", response_model=AlertResponse)
def declining_alerts(request: AlertRequest, db: Session = Depends(get_db)) -> AlertResponse:
    try:
        service = ForecastService(db)
        return service.declining_alerts(
            entity_type=request.entity_type,
            target_metric=request.target_metric,
            horizon_weeks=request.horizon_weeks,
            declining_threshold_pct=request.declining_threshold_pct,
            top_k=request.top_k,
            cluster_id=request.cluster_id,
            cluster_label_contains=request.cluster_label_contains,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/agent/query", response_model=AgentResponse)
def agent_query(request: AgentQueryRequest, db: Session = Depends(get_db)) -> AgentResponse:
    try:
        return AgentRouterService(db).handle_query(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
