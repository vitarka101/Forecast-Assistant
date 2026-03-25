from pydantic import BaseModel, Field

from app.schemas.common import EntityType, TargetMetric


class TrainPipelineRequest(BaseModel):
    data_path: str | None = None
    product_clusters: int = Field(default=4, ge=1)
    customer_clusters: int = Field(default=4, ge=1)
    min_history_weeks: int = Field(default=8, ge=2)
    max_entities_for_dtw: int = Field(default=250, ge=10)
    lag_weeks: int = Field(default=8, ge=2)


class ForecastRequest(BaseModel):
    entity_type: EntityType
    entity_id: str
    horizon_weeks: int = Field(default=4, ge=1, le=26)
    target_metric: TargetMetric = TargetMetric.REVENUE
    price_multiplier: float = Field(default=1.0, gt=0.0, le=10.0)


class CompareRequest(BaseModel):
    entity_type: EntityType
    entity_ids: list[str] = Field(min_length=2, max_length=10)
    horizon_weeks: int = Field(default=4, ge=1, le=26)
    target_metric: TargetMetric = TargetMetric.REVENUE
    price_multiplier: float = Field(default=1.0, gt=0.0, le=10.0)


class AlertRequest(BaseModel):
    entity_type: EntityType
    cluster_id: str | None = None
    cluster_label_contains: str | None = None
    horizon_weeks: int = Field(default=4, ge=1, le=26)
    target_metric: TargetMetric = TargetMetric.REVENUE
    declining_threshold_pct: float = Field(default=0.1, ge=0.0, le=1.0)
    top_k: int = Field(default=10, ge=1, le=100)


class AgentQueryRequest(BaseModel):
    query: str
    default_entity_type: EntityType | None = None
    horizon_weeks: int = Field(default=4, ge=1, le=26)
    target_metric: TargetMetric = TargetMetric.REVENUE

