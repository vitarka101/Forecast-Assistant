from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.common import AgentIntent, EntityType, TargetMetric


class ForecastPoint(BaseModel):
    week_start: date
    value: float


class HistoryPoint(BaseModel):
    week_start: date
    value: float


class HistoryResponse(BaseModel):
    entity_type: EntityType
    entity_id: str
    target_metric: TargetMetric
    lookback_weeks: int
    points: list[HistoryPoint]
    summary: str


class EntityContextResponse(BaseModel):
    entity_type: EntityType
    entity_id: str
    short_label: str | None = None
    note: str | None = None


class ForecastResponse(BaseModel):
    entity_type: EntityType
    entity_id: str
    cluster_id: str
    cluster_label: str
    entity_status: str
    forecast_strategy: str
    target_metric: TargetMetric
    horizon_weeks: int
    baseline_recent_average: float
    first_week_change_pct: float | None
    total_forecast: float
    forecast: list[ForecastPoint]
    narrative: str
    data_quality_summary: str | None = None
    data_issue_codes: list[str] = Field(default_factory=list)


class ComparisonResponse(BaseModel):
    entity_type: EntityType
    target_metric: TargetMetric
    comparison_summary: str
    forecasts: list[ForecastResponse]


class AlertItem(BaseModel):
    entity_id: str
    cluster_id: str
    cluster_label: str
    recent_average: float
    forecast_average: float
    decline_pct: float


class AlertResponse(BaseModel):
    entity_type: EntityType
    target_metric: TargetMetric
    alerts: list[AlertItem]
    summary: str


class EntityTrainingSummary(BaseModel):
    entity_type: EntityType
    weekly_rows: int
    mapped_entities: int
    clusters_built: int
    models_trained: int
    issue_only_entities: int = 0
    sparse_history_entities: int = 0
    largest_cluster_share: float | None = None


class TrainPipelineResponse(BaseModel):
    data_path: str
    transactions_loaded: int
    entities: list[EntityTrainingSummary]
    message: str


class AgentResponse(BaseModel):
    intent: AgentIntent
    tool_used: str
    summary: str
    payload: dict[str, Any]
