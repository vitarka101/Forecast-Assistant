import json
from pathlib import Path

import httpx
from pydantic import ValidationError

from app.core.config import settings
from app.schemas.common import EntityType, TargetMetric
from app.services.llm.base import (
    HeuristicRouterProvider,
    RouterDecision,
    RouterDecisionPayload,
    merge_router_decisions,
)


PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "system_prompt.txt"


class OllamaRouterProvider:
    def __init__(self) -> None:
        if not settings.llm_model:
            raise ValueError("LLM_MODEL must be set when LLM_PROVIDER=Ollama")
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.llm_model
        self.fallback = HeuristicRouterProvider()
        self.system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    def decide(
        self,
        query: str,
        default_entity_type: EntityType | None,
        horizon_weeks: int,
        target_metric: TargetMetric,
    ) -> RouterDecision:
        heuristic = self.fallback.decide(query, default_entity_type, horizon_weeks, target_metric)
        schema = RouterDecisionPayload.model_json_schema()
        payload = {
            "model": self.model,
            "stream": False,
            "format": schema,
            "options": {"temperature": 0},
            "messages": [
                {
                    "role": "system",
                    "content": self._system_message(schema),
                },
                {
                    "role": "user",
                    "content": self._user_message(query, default_entity_type, horizon_weeks, target_metric),
                },
            ],
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = ""
            try:
                detail = exc.response.text.strip()
            except Exception:
                detail = ""
            suffix = f" Ollama response: {detail}" if detail else ""
            raise ValueError(
                f"Ollama returned HTTP {exc.response.status_code} for model '{self.model}'.{suffix}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ValueError(
                f"Failed to reach Ollama at {self.base_url}. Ensure Ollama is running and the base URL is correct."
            ) from exc

        content = response.json().get("message", {}).get("content", "").strip()
        if not content:
            return heuristic

        parsed = self._parse_payload(content)
        return merge_router_decisions(parsed, heuristic)

    def _system_message(self, schema: dict) -> str:
        return (
            f"{self.system_prompt}\n\n"
            "Return only JSON that matches this schema exactly.\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            "Interpretation rules:\n"
            "- If the query mentions product or stock code, entity_type must be 'product'.\n"
            "- If the query mentions customer or consumer, entity_type must be 'customer'.\n"
            "- Use stock_code values as product identifiers.\n"
            "- Use customer_id values as customer identifiers.\n"
            "- For compare queries, include two or more entity_ids.\n"
            "- For price scenarios, output price_multiplier like 1.05 for +5% or 0.95 for -5%.\n"
            "- If the query requests at-risk or declining entities, use intent='alerts'.\n"
            "- If a field is unknown, use null or an empty list instead of inventing data."
        )

    @staticmethod
    def _user_message(
        query: str,
        default_entity_type: EntityType | None,
        horizon_weeks: int,
        target_metric: TargetMetric,
    ) -> str:
        return (
            f"User query: {query}\n"
            f"default_entity_type: {default_entity_type.value if default_entity_type else 'null'}\n"
            f"default_horizon_weeks: {horizon_weeks}\n"
            f"default_target_metric: {target_metric.value}\n"
        )

    @staticmethod
    def _parse_payload(content: str) -> RouterDecisionPayload:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        try:
            return RouterDecisionPayload.model_validate_json(cleaned)
        except ValidationError:
            try:
                return RouterDecisionPayload.model_validate(json.loads(cleaned))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(f"Ollama returned invalid router JSON: {content}") from exc
