from collections import Counter

import numpy as np
import pandas as pd

from app.schemas.common import EntityType


STATUS_OK = "ok"
STATUS_SPARSE_HISTORY = "sparse_history"
STATUS_DATA_ISSUE = "data_issue"

STRATEGY_CLUSTER_MODEL = "cluster_gradient_boosting"
STRATEGY_SIMILAR_PEERS = "similar_peer_blend"
STRATEGY_ZERO_FORECAST = "zero_from_data_quality"

ISSUE_LABELS = {
    "duplicate_row": "duplicate rows",
    "cancelled_invoice": "cancelled invoices",
    "non_positive_quantity": "non-positive quantities",
    "non_positive_price": "non-positive prices",
    "missing_invoice_date": "missing invoice dates",
    "missing_stock_code": "missing stock codes",
    "missing_customer_id": "missing customer IDs",
}


def build_entity_catalog(
    transactions: pd.DataFrame,
    weekly: pd.DataFrame,
    assignments: pd.DataFrame,
    entity_type: EntityType,
    min_history_weeks: int,
    lag_weeks: int,
    target_column: str,
    peer_count: int = 5,
) -> pd.DataFrame:
    id_column = "stock_code" if entity_type == EntityType.PRODUCT else "customer_id"
    valid_column = "valid_for_product" if entity_type == EntityType.PRODUCT else "valid_for_customer"
    noun = entity_type.value

    entity_rows = transactions[transactions[id_column].notna()].copy()
    if entity_rows.empty:
        return pd.DataFrame(
            columns=[
                "entity_id",
                "status",
                "forecast_strategy",
                "issue_summary",
                "issue_codes",
                "valid_transaction_count",
                "issue_transaction_count",
                "active_weeks",
                "total_weeks",
                "cluster_id",
                "cluster_label",
                "nearest_peer_ids",
                "metadata_json",
            ]
        )

    catalog = (
        entity_rows.groupby(id_column)
        .agg(
            valid_transaction_count=(valid_column, "sum"),
            total_transaction_count=(valid_column, "size"),
        )
        .rename_axis("entity_id")
        .reset_index()
    )
    catalog["valid_transaction_count"] = catalog["valid_transaction_count"].astype(int)
    catalog["issue_transaction_count"] = (
        catalog["total_transaction_count"] - catalog["valid_transaction_count"]
    ).astype(int)

    issue_payload = (
        entity_rows.groupby(id_column)["row_issue_codes"]
        .apply(_issue_counts)
        .reset_index(name="issue_payload")
        .rename(columns={id_column: "entity_id"})
    )
    catalog = catalog.merge(issue_payload, on="entity_id", how="left")
    catalog["issue_payload"] = catalog["issue_payload"].apply(lambda payload: payload if isinstance(payload, dict) else {})

    weekly_stats = _weekly_stats(weekly, target_column)
    catalog = catalog.merge(weekly_stats, on="entity_id", how="left")
    catalog["active_weeks"] = catalog["active_weeks"].fillna(0).astype(int)
    catalog["total_weeks"] = catalog["total_weeks"].fillna(0).astype(int)
    catalog["recent_average"] = catalog["recent_average"].fillna(0.0)
    catalog["target_total"] = catalog["target_total"].fillna(0.0)
    catalog["avg_unit_price_mean"] = catalog["avg_unit_price_mean"].fillna(0.0)

    assignment_view = assignments[["entity_id", "cluster_id", "label"]].rename(columns={"label": "cluster_label"})
    catalog = catalog.merge(assignment_view, on="entity_id", how="left")
    catalog = catalog.drop_duplicates(subset=["entity_id"], keep="first").reset_index(drop=True)

    catalog["status"] = np.where(
        catalog["valid_transaction_count"] == 0,
        STATUS_DATA_ISSUE,
        np.where(catalog["active_weeks"] < min_history_weeks, STATUS_SPARSE_HISTORY, STATUS_OK),
    )
    catalog["forecast_strategy"] = np.select(
        [
            catalog["status"] == STATUS_DATA_ISSUE,
            catalog["status"] == STATUS_SPARSE_HISTORY,
        ],
        [
            STRATEGY_ZERO_FORECAST,
            STRATEGY_SIMILAR_PEERS,
        ],
        default=STRATEGY_CLUSTER_MODEL,
    )

    peer_lookup = _precompute_sparse_peers(catalog, weekly, target_column, lag_weeks, min_history_weeks, peer_count)
    catalog["nearest_peer_ids"] = catalog["entity_id"].map(lambda entity_id: peer_lookup.get(entity_id, {}).get("ids", []))
    catalog["metadata_json"] = catalog.apply(
        lambda row: {
            "recent_average": round(float(row["recent_average"]), 4),
            "target_total": round(float(row["target_total"]), 4),
            "avg_unit_price_mean": round(float(row["avg_unit_price_mean"]), 4),
            "peer_weights": peer_lookup.get(row["entity_id"], {}).get("weights", []),
            "issue_counts": row["issue_payload"],
        },
        axis=1,
    )
    catalog["issue_codes"] = catalog["issue_payload"].map(lambda payload: sorted(payload.keys()))
    catalog["issue_summary"] = catalog.apply(
        lambda row: _issue_summary(
            issue_counts=row["issue_payload"],
            status=row["status"],
            active_weeks=int(row["active_weeks"]),
            min_history_weeks=min_history_weeks,
            noun=noun,
        ),
        axis=1,
    )

    return catalog[
        [
            "entity_id",
            "status",
            "forecast_strategy",
            "issue_summary",
            "issue_codes",
            "valid_transaction_count",
            "issue_transaction_count",
            "active_weeks",
            "total_weeks",
            "cluster_id",
            "cluster_label",
            "nearest_peer_ids",
            "metadata_json",
        ]
    ].sort_values("entity_id").reset_index(drop=True)


def _weekly_stats(weekly: pd.DataFrame, target_column: str) -> pd.DataFrame:
    if weekly.empty:
        return pd.DataFrame(
            columns=["entity_id", "active_weeks", "total_weeks", "recent_average", "target_total", "avg_unit_price_mean"]
        )

    return (
        weekly.sort_values(["entity_id", "week_start"])
        .groupby("entity_id")
        .agg(
            active_weeks=(target_column, lambda values: int((values > 0).sum())),
            total_weeks=("week_start", "nunique"),
            recent_average=(target_column, lambda values: float(values.tail(4).mean()) if len(values) else 0.0),
            target_total=(target_column, "sum"),
            avg_unit_price_mean=("avg_unit_price", "mean"),
        )
        .reset_index()
    )


def _issue_counts(series: pd.Series) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for codes in series:
        counter.update(codes)
    return dict(counter)


def _issue_summary(
    issue_counts: dict[str, int],
    status: str,
    active_weeks: int,
    min_history_weeks: int,
    noun: str,
) -> str | None:
    messages: list[str] = []
    if issue_counts:
        details = ", ".join(
            f"{count} {ISSUE_LABELS.get(code, code.replace('_', ' '))}"
            for code, count in sorted(issue_counts.items(), key=lambda item: (-item[1], item[0]))
        )
        messages.append(f"Retained rows were excluded from modeling because the source data contained {details}.")

    if status == STATUS_DATA_ISSUE:
        messages.append("No clean history remained after notebook-style cleaning, so the forecast is fixed at 0.")
    elif status == STATUS_SPARSE_HISTORY:
        messages.append(
            f"Only {active_weeks} active weeks remained after cleaning, below the {min_history_weeks}-week threshold, "
            f"so the forecast is blended from similar {noun}s."
        )

    return " ".join(messages) or None


def _precompute_sparse_peers(
    catalog: pd.DataFrame,
    weekly: pd.DataFrame,
    target_column: str,
    lag_weeks: int,
    min_history_weeks: int,
    peer_count: int,
) -> dict[str, dict[str, list[float] | list[str]]]:
    if weekly.empty or catalog.empty:
        return {}

    feature_frame = _peer_feature_frame(weekly, target_column, lag_weeks)
    if feature_frame.empty:
        return {}

    dense = catalog[catalog["active_weeks"] >= min_history_weeks].merge(feature_frame, on="entity_id", how="inner")
    sparse = catalog[catalog["status"] == STATUS_SPARSE_HISTORY].merge(feature_frame, on="entity_id", how="inner")
    if dense.empty or sparse.empty:
        return {}

    feature_columns = [
        column
        for column in feature_frame.columns
        if column.startswith("recent_shape_")
        or column in {"active_weeks_feature", "recent_level", "recent_volatility", "recent_density"}
    ]
    dense_by_cluster = {cluster_id: frame.reset_index(drop=True) for cluster_id, frame in dense.groupby("cluster_id")}
    global_dense = dense.reset_index(drop=True)

    peer_lookup: dict[str, dict[str, list[float] | list[str]]] = {}
    for row in sparse.itertuples(index=False):
        candidates = dense_by_cluster.get(row.cluster_id)
        if candidates is None or candidates.empty:
            candidates = global_dense
        if candidates.empty:
            continue

        target_vector = np.array([getattr(row, column) for column in feature_columns], dtype=float)
        candidate_matrix = candidates[feature_columns].to_numpy(dtype=float)
        distances = np.linalg.norm(candidate_matrix - target_vector, axis=1)
        order = np.argsort(distances)[:peer_count]
        top = candidates.iloc[order]
        inverse = 1.0 / np.maximum(distances[order], 1e-6)
        weights = (inverse / inverse.sum()).tolist()
        peer_lookup[row.entity_id] = {
            "ids": top["entity_id"].tolist(),
            "weights": [round(float(weight), 6) for weight in weights],
        }
    return peer_lookup


def _peer_feature_frame(weekly: pd.DataFrame, target_column: str, lag_weeks: int) -> pd.DataFrame:
    rows: list[dict] = []
    for entity_id, group in weekly.sort_values(["entity_id", "week_start"]).groupby("entity_id"):
        series = group[target_column].to_numpy(dtype=float)
        positive = series[series > 0]
        recent = positive[-lag_weeks:] if len(positive) else np.array([], dtype=float)
        padded = np.zeros(lag_weeks, dtype=float)
        if len(recent):
            padded[-len(recent) :] = recent
        scale = float(np.max(padded)) if np.max(padded) > 0 else 1.0
        normalized = padded / scale
        row = {
            "entity_id": entity_id,
            "active_weeks_feature": float((series > 0).sum()),
            "recent_level": float(recent.mean()) if len(recent) else 0.0,
            "recent_volatility": float(recent.std()) if len(recent) > 1 else 0.0,
            "recent_density": float(len(recent) / max(lag_weeks, 1)),
        }
        for index, value in enumerate(normalized, start=1):
            row[f"recent_shape_{index}"] = float(value)
        rows.append(row)

    return pd.DataFrame(rows)
