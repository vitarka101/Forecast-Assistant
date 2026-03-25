from enum import Enum


class EntityType(str, Enum):
    PRODUCT = "product"
    CUSTOMER = "customer"


class TargetMetric(str, Enum):
    REVENUE = "revenue"
    QUANTITY = "quantity"


class AgentIntent(str, Enum):
    FORECAST = "forecast"
    COMPARE = "compare"
    SCENARIO = "scenario"
    ALERTS = "alerts"

