import numpy as np
import pandas as pd

from app.schemas.common import EntityType


METRIC_COLUMNS = ["revenue", "quantity", "invoice_count", "counterparties", "avg_unit_price"]


def build_entity_weekly_metrics(transactions: pd.DataFrame, entity_type: EntityType) -> pd.DataFrame:
    if entity_type == EntityType.PRODUCT:
        frame = transactions[transactions["valid_for_product"]].dropna(subset=["stock_code", "week_start"]).copy()
        frame["entity_id"] = frame["stock_code"]
        counterpart_column = "customer_id"
    else:
        frame = transactions[transactions["valid_for_customer"]].dropna(subset=["customer_id", "week_start"]).copy()
        frame["entity_id"] = frame["customer_id"]
        counterpart_column = "stock_code"

    if frame.empty:
        return pd.DataFrame(columns=["entity_id", "week_start", *METRIC_COLUMNS, "entity_type"])

    weekly = (
        frame.groupby(["entity_id", "week_start"], as_index=False)
        .agg(
            revenue=("revenue", "sum"),
            quantity=("quantity", "sum"),
            invoice_count=("invoice_no", "nunique"),
            counterparties=(counterpart_column, lambda values: values.dropna().nunique()),
        )
        .sort_values(["entity_id", "week_start"])
    )

    weekly["avg_unit_price"] = np.where(
        weekly["quantity"] > 0,
        weekly["revenue"] / weekly["quantity"],
        0.0,
    )
    weekly = _complete_week_grid(weekly)
    weekly["entity_type"] = entity_type.value
    return weekly


def _complete_week_grid(weekly: pd.DataFrame) -> pd.DataFrame:
    if weekly.empty:
        return weekly.copy()

    all_weeks = pd.date_range(weekly["week_start"].min(), weekly["week_start"].max(), freq="W-MON")
    completed: list[pd.DataFrame] = []
    for entity_id, group in weekly.groupby("entity_id"):
        block = (
            group.set_index("week_start")[METRIC_COLUMNS]
            .reindex(all_weeks, fill_value=0.0)
            .rename_axis("week_start")
            .reset_index()
        )
        block["entity_id"] = entity_id
        completed.append(block)

    result = pd.concat(completed, ignore_index=True)
    ordered = ["entity_id", "week_start", *METRIC_COLUMNS]
    return result[ordered].sort_values(["entity_id", "week_start"]).reset_index(drop=True)


def build_shape_matrix(
    weekly: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    pivot = (
        weekly.pivot(index="entity_id", columns="week_start", values=target_column)
        .fillna(0.0)
        .sort_index(axis=1)
    )
    return pivot


def normalize_shape_matrix(shape_matrix: pd.DataFrame) -> pd.DataFrame:
    values = shape_matrix.to_numpy(dtype=float)
    means = values.mean(axis=1, keepdims=True)
    stds = values.std(axis=1, keepdims=True)
    normalized = np.where(stds > 0, (values - means) / stds, values - means)
    return pd.DataFrame(normalized, index=shape_matrix.index, columns=shape_matrix.columns)
