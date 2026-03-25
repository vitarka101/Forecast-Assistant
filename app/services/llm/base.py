import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from app.core.config import settings
from app.schemas.common import AgentIntent, EntityType, TargetMetric


@dataclass
class RouterDecision:
    intent: AgentIntent
    entity_type: EntityType | None = None
    entity_ids: list[str] = field(default_factory=list)
    cluster_label_contains: str | None = None
    price_multiplier: float = 1.0
    horizon_weeks: int = 4
    target_metric: TargetMetric = TargetMetric.REVENUE


class RouterDecisionPayload(BaseModel):
    intent: AgentIntent = AgentIntent.FORECAST
    entity_type: EntityType | None = None
    entity_ids: list[str] = Field(default_factory=list)
    cluster_label_contains: str | None = None
    price_multiplier: float | None = Field(default=None, gt=0.0, le=10.0)
    horizon_weeks: int | None = Field(default=None, ge=1, le=26)
    target_metric: TargetMetric | None = None


class HeuristicRouterProvider:
    entity_pattern = re.compile(
        r"(product|stock(?:_| )?code|customer|consumer)\s*[#:]*\s*([A-Za-z0-9\-\.]+)",
        re.IGNORECASE,
    )

    def decide(
        self,
        query: str,
        default_entity_type: EntityType | None,
        horizon_weeks: int,
        target_metric: TargetMetric,
    ) -> RouterDecision:
        lowered = query.lower()
        intent = AgentIntent.FORECAST
        if "compare" in lowered or " vs " in lowered:
            intent = AgentIntent.COMPARE
        elif "declin" in lowered or "at-risk" in lowered or "at risk" in lowered or "alert" in lowered:
            intent = AgentIntent.ALERTS
        elif "price" in lowered or "scenario" in lowered:
            intent = AgentIntent.SCENARIO

        entity_ids: list[str] = []
        detected_type = default_entity_type
        for label, entity_id in self.entity_pattern.findall(query):
            normalized_type = EntityType.PRODUCT if label.lower().startswith(("product", "stock")) else EntityType.CUSTOMER
            detected_type = normalized_type
            entity_ids.append(entity_id.upper())

        if not entity_ids:
            entity_ids = _extract_fallback_ids(query, detected_type)

        if detected_type is None and entity_ids:
            detected_type = EntityType.PRODUCT if any(char.isalpha() for char in entity_ids[0]) else EntityType.CUSTOMER

        cluster_label = None
        cluster_match = re.search(r"'([^']+)' cluster|\"([^\"]+)\" cluster", query, re.IGNORECASE)
        if cluster_match:
            cluster_label = next(group for group in cluster_match.groups() if group)
        elif "at-risk" in lowered or "at risk" in lowered:
            cluster_label = "declining"

        price_multiplier = 1.0
        pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", query)
        if pct_match:
            pct = float(pct_match.group(1)) / 100.0
            if "decrease" in lowered or "drop" in lowered or "reduce" in lowered:
                price_multiplier = max(0.01, 1.0 - pct)
            else:
                price_multiplier = 1.0 + pct

        return RouterDecision(
            intent=intent,
            entity_type=detected_type,
            entity_ids=entity_ids,
            cluster_label_contains=cluster_label,
            price_multiplier=price_multiplier,
            horizon_weeks=horizon_weeks,
            target_metric=target_metric,
        )


def _extract_fallback_ids(query: str, entity_type: EntityType | None) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9\-\.]+", query)
    if entity_type == EntityType.CUSTOMER:
        candidates = [token for token in tokens if token.isdigit()]
    elif entity_type == EntityType.PRODUCT:
        candidates = [token.upper() for token in tokens if any(char.isalpha() for char in token)]
    else:
        candidates = [token.upper() for token in tokens if len(token) >= 4]
    stopwords = {"COMPARE", "FORECAST", "PRICE", "CLUSTER", "CUSTOMER", "PRODUCT", "STOCK", "CODE"}
    return [token for token in candidates if token.upper() not in stopwords][:4]


def merge_router_decisions(
    primary: RouterDecisionPayload,
    fallback: RouterDecision,
) -> RouterDecision:
    entity_ids = [entity_id.strip().upper() for entity_id in primary.entity_ids if entity_id and entity_id.strip()]
    if not entity_ids:
        entity_ids = fallback.entity_ids

    return RouterDecision(
        intent=primary.intent or fallback.intent,
        entity_type=primary.entity_type or fallback.entity_type,
        entity_ids=entity_ids,
        cluster_label_contains=primary.cluster_label_contains or fallback.cluster_label_contains,
        price_multiplier=primary.price_multiplier if primary.price_multiplier is not None else fallback.price_multiplier,
        horizon_weeks=primary.horizon_weeks or fallback.horizon_weeks,
        target_metric=primary.target_metric or fallback.target_metric,
    )


def get_router_provider():
    provider = settings.llm_provider.lower().strip()
    if provider == "heuristic":
        return HeuristicRouterProvider()
    if provider == "ollama":
        from app.services.llm.ollama import OllamaRouterProvider

        return OllamaRouterProvider()
    raise ValueError(f"Unsupported llm_provider: {settings.llm_provider}")
