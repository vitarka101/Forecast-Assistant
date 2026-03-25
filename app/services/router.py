from sqlalchemy.orm import Session

from app.schemas.common import AgentIntent
from app.schemas.requests import AgentQueryRequest
from app.schemas.responses import AgentResponse
from app.services.forecasting import ForecastService
from app.services.llm.base import get_router_provider


class AgentRouterService:
    def __init__(self, db: Session):
        self.db = db
        self.forecasts = ForecastService(db)
        self.provider = get_router_provider()

    def handle_query(self, request: AgentQueryRequest) -> AgentResponse:
        decision = self.provider.decide(
            query=request.query,
            default_entity_type=request.default_entity_type,
            horizon_weeks=request.horizon_weeks,
            target_metric=request.target_metric,
        )

        if decision.entity_type is None:
            raise ValueError("Could not infer entity type from the query. Specify product or customer.")

        if decision.intent == AgentIntent.COMPARE:
            if len(decision.entity_ids) < 2:
                raise ValueError("Comparison queries need at least two entity IDs.")
            payload = self.forecasts.compare_entities(
                entity_type=decision.entity_type,
                entity_ids=decision.entity_ids[:2],
                horizon_weeks=decision.horizon_weeks,
                target_metric=decision.target_metric,
                price_multiplier=decision.price_multiplier,
            )
            return AgentResponse(
                intent=decision.intent,
                tool_used="compare_entities",
                summary=payload.comparison_summary,
                payload=payload.model_dump(),
            )

        if decision.intent == AgentIntent.ALERTS:
            payload = self.forecasts.declining_alerts(
                entity_type=decision.entity_type,
                target_metric=decision.target_metric,
                horizon_weeks=decision.horizon_weeks,
                declining_threshold_pct=0.1,
                top_k=10,
                cluster_label_contains=decision.cluster_label_contains or "declining",
            )
            return AgentResponse(
                intent=decision.intent,
                tool_used="declining_alerts",
                summary=payload.summary,
                payload=payload.model_dump(),
            )

        if not decision.entity_ids:
            raise ValueError("No entity ID found in the query.")

        forecast = self.forecasts.forecast_entity(
            entity_type=decision.entity_type,
            entity_id=decision.entity_ids[0],
            horizon_weeks=decision.horizon_weeks,
            target_metric=decision.target_metric,
            price_multiplier=decision.price_multiplier,
        )
        tool_name = "scenario_forecast" if decision.intent == AgentIntent.SCENARIO else "forecast_entity"
        return AgentResponse(
            intent=decision.intent,
            tool_used=tool_name,
            summary=forecast.narrative,
            payload=forecast.model_dump(),
        )

