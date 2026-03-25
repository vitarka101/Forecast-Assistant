from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sqlalchemy.orm import Session

from app.core.config import settings
from app.repositories.catalog import (
    find_cluster_profiles_by_label,
    get_assignment,
    get_cluster_profile,
    get_entity_catalog_record,
    get_model_record,
    list_assignments_for_cluster,
)
from app.schemas.common import EntityType, TargetMetric
from app.schemas.responses import (
    AlertItem,
    AlertResponse,
    ComparisonResponse,
    ForecastPoint,
    ForecastResponse,
    HistoryPoint,
    HistoryResponse,
)
from app.services.entity_catalog import (
    STATUS_DATA_ISSUE,
    STATUS_OK,
    STATUS_SPARSE_HISTORY,
    STRATEGY_CLUSTER_MODEL,
    STRATEGY_SIMILAR_PEERS,
    STRATEGY_ZERO_FORECAST,
)


GB_CANDIDATES = [
    (
        "GradientBoostingRegressor.GB-Normal",
        {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "max_features": "sqrt",
            "min_samples_leaf": 3,
            "min_samples_split": 6,
            "subsample": 0.8,
            "random_state": settings.random_state,
        },
    ),
    (
        "GradientBoostingRegressor.GB-Volatile",
        {
            "n_estimators": 300,
            "learning_rate": 0.02,
            "max_depth": 3,
            "max_features": "sqrt",
            "min_samples_leaf": 5,
            "min_samples_split": 10,
            "subsample": 0.7,
            "random_state": settings.random_state,
        },
    ),
]


@dataclass
class TrainingResult:
    model_name: str
    artifact_path: str
    feature_names: list[str]
    metrics: dict


def weekly_snapshot_path(entity_type: EntityType) -> Path:
    return Path(settings.artifacts_dir) / f"{entity_type.value}_weekly_metrics.csv"


def model_artifact_dir(entity_type: EntityType) -> Path:
    path = Path(settings.artifacts_dir) / "models" / entity_type.value
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_weekly_snapshot(entity_type: EntityType, weekly: pd.DataFrame) -> Path:
    path = weekly_snapshot_path(entity_type)
    weekly.to_csv(path, index=False)
    return path


def load_weekly_snapshot(entity_type: EntityType) -> pd.DataFrame:
    path = weekly_snapshot_path(entity_type)
    if not path.exists():
        raise FileNotFoundError(f"Missing weekly snapshot for {entity_type.value}. Run training first.")
    weekly = pd.read_csv(
        path,
        parse_dates=["week_start"],
        dtype={"entity_id": "string", "entity_type": "string"},
        low_memory=False,
    )
    weekly["entity_id"] = (
        weekly["entity_id"].fillna("").astype(str).str.strip().str.removesuffix(".0").str.upper()
    )
    if "entity_type" in weekly.columns:
        weekly["entity_type"] = weekly["entity_type"].fillna("").astype(str).str.strip().str.lower()
    return weekly


def train_cluster_model(
    weekly: pd.DataFrame,
    assignments: pd.DataFrame,
    entity_type: EntityType,
    cluster_id: str,
    target_metric: TargetMetric,
    lag_weeks: int,
) -> TrainingResult | None:
    joined = weekly.merge(assignments[["entity_id", "cluster_id"]], on="entity_id", how="inner")
    cluster_frame = joined[joined["cluster_id"] == cluster_id].copy()
    if cluster_frame.empty:
        return None

    training_frame, feature_names = _build_training_frame(cluster_frame, target_metric.value, lag_weeks)
    model, model_name = _fit_model(training_frame, feature_names)
    metrics = _evaluate_model(model, training_frame, feature_names)
    cluster_baseline = _recent_positive_mean(cluster_frame[target_metric.value])

    artifact_path = model_artifact_dir(entity_type) / f"{cluster_id}_{target_metric.value}.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "target_metric": target_metric.value,
            "lag_weeks": lag_weeks,
            "entity_type": entity_type.value,
            "cluster_id": cluster_id,
            "cluster_baseline": cluster_baseline,
        },
        artifact_path,
    )
    return TrainingResult(
        model_name=model_name,
        artifact_path=str(artifact_path),
        feature_names=feature_names,
        metrics=metrics,
    )


def _build_training_frame(
    cluster_frame: pd.DataFrame,
    target_column: str,
    lag_weeks: int,
) -> tuple[pd.DataFrame, list[str]]:
    parts: list[pd.DataFrame] = []
    for _, group in cluster_frame.groupby("entity_id"):
        block = group.sort_values("week_start").copy()
        for lag in range(1, lag_weeks + 1):
            block[f"{target_column}_lag_{lag}"] = block[target_column].shift(lag)
        block["rolling_mean_4"] = block[target_column].shift(1).rolling(4).mean()
        block["rolling_mean_8"] = block[target_column].shift(1).rolling(8).mean()
        block["rolling_std_4"] = block[target_column].shift(1).rolling(4).std()
        block["avg_unit_price_lag_1"] = block["avg_unit_price"].shift(1)
        block["invoice_count_lag_1"] = block["invoice_count"].shift(1)
        block["counterparties_lag_1"] = block["counterparties"].shift(1)
        week_number = block["week_start"].dt.isocalendar().week.astype(int)
        block["week_sin"] = np.sin(2 * np.pi * week_number / 52.0)
        block["week_cos"] = np.cos(2 * np.pi * week_number / 52.0)
        block["target_next"] = block[target_column].shift(-1)
        parts.append(block)

    if not parts:
        return pd.DataFrame(), _feature_names(target_column, lag_weeks)

    training = pd.concat(parts, ignore_index=True)
    feature_names = _feature_names(target_column, lag_weeks)
    training[feature_names] = training[feature_names].fillna(0.0)
    training = training.dropna(subset=["target_next"]).reset_index(drop=True)
    return training, feature_names


def _feature_names(target_column: str, lag_weeks: int) -> list[str]:
    return [f"{target_column}_lag_{lag}" for lag in range(1, lag_weeks + 1)] + [
        "rolling_mean_4",
        "rolling_mean_8",
        "rolling_std_4",
        "avg_unit_price_lag_1",
        "invoice_count_lag_1",
        "counterparties_lag_1",
        "week_sin",
        "week_cos",
    ]


def _fit_model(training: pd.DataFrame, feature_names: list[str]):
    if training.empty:
        model = DummyRegressor(strategy="constant", constant=0.0)
        model.fit(pd.DataFrame([[0.0] * len(feature_names)], columns=feature_names), [0.0])
        return model, "DummyRegressor.Zero"

    if len(training) < 12:
        model = DummyRegressor(strategy="mean")
        model.fit(training[feature_names], training["target_next"])
        return model, "DummyRegressor.Mean"

    candidate_specs = list(GB_CANDIDATES)
    if len(training) >= 48:
        candidate_specs.append(
            (
                "RandomForestRegressor",
                {
                    "n_estimators": 250,
                    "min_samples_leaf": 2,
                    "random_state": settings.random_state,
                    "n_jobs": -1,
                },
            )
        )

    best_name = "DummyRegressor.Mean"
    best_score = float("inf")
    best_factory = None
    for candidate_name, params in candidate_specs:
        factory = (
            (lambda params=params: GradientBoostingRegressor(**params))
            if candidate_name.startswith("GradientBoostingRegressor")
            else (lambda params=params: RandomForestRegressor(**params))
        )
        score = _temporal_validation_mape(factory, training, feature_names)
        if score < best_score:
            best_name = candidate_name
            best_score = score
            best_factory = factory

    if best_factory is None:
        model = DummyRegressor(strategy="mean")
    else:
        model = best_factory()
    model.fit(training[feature_names], training["target_next"])
    return model, best_name


def _temporal_validation_mape(factory, training: pd.DataFrame, feature_names: list[str]) -> float:
    unique_weeks = sorted(training["week_start"].unique())
    if len(unique_weeks) < 6:
        model = factory()
        model.fit(training[feature_names], training["target_next"])
        predicted = np.maximum(model.predict(training[feature_names]), 0.0)
        return _metric_bundle(training["target_next"].to_numpy(), predicted)["mape_pct"]

    cutoff_index = max(1, int(len(unique_weeks) * 0.8))
    cutoff_week = unique_weeks[cutoff_index]
    train_split = training[training["week_start"] < cutoff_week]
    validation = training[training["week_start"] >= cutoff_week]
    if train_split.empty or validation.empty:
        train_split = training
        validation = training

    model = factory()
    model.fit(train_split[feature_names], train_split["target_next"])
    predicted = np.maximum(model.predict(validation[feature_names]), 0.0)
    return _metric_bundle(validation["target_next"].to_numpy(), predicted)["mape_pct"]


def _evaluate_model(model, training: pd.DataFrame, feature_names: list[str]) -> dict:
    if training.empty:
        return {"mae": 0.0, "rmse": 0.0, "mape_pct": 0.0}

    unique_weeks = sorted(training["week_start"].unique())
    if len(unique_weeks) < 6:
        predictions = np.maximum(model.predict(training[feature_names]), 0.0)
        return _metric_bundle(training["target_next"].to_numpy(), predictions)

    cutoff_index = max(1, int(len(unique_weeks) * 0.8))
    cutoff_week = unique_weeks[cutoff_index]
    validation = training[training["week_start"] >= cutoff_week]
    if validation.empty:
        validation = training
    predictions = np.maximum(model.predict(validation[feature_names]), 0.0)
    return _metric_bundle(validation["target_next"].to_numpy(), predictions)


def _metric_bundle(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mask = actual != 0
    if np.any(mask):
        mape = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)
    else:
        mape = 0.0
    return {"mae": mae, "rmse": rmse, "mape_pct": mape}


class ForecastService:
    def __init__(self, db: Session):
        self.db = db

    def entity_history(
        self,
        entity_type: EntityType,
        entity_id: str,
        target_metric: TargetMetric,
        lookback_weeks: int,
    ) -> HistoryResponse:
        weekly = load_weekly_snapshot(entity_type)
        history = weekly[weekly["entity_id"] == entity_id].sort_values("week_start").copy()
        if history.empty:
            catalog_record = get_entity_catalog_record(self.db, entity_type, entity_id)
            summary = (
                catalog_record.issue_summary
                if catalog_record is not None and catalog_record.issue_summary
                else f"No clean weekly actuals are available for {entity_type.value} {entity_id}."
            )
            return HistoryResponse(
                entity_type=entity_type,
                entity_id=entity_id,
                target_metric=target_metric,
                lookback_weeks=lookback_weeks,
                points=[],
                summary=summary,
            )

        history_tail = history.tail(lookback_weeks)
        return HistoryResponse(
            entity_type=entity_type,
            entity_id=entity_id,
            target_metric=target_metric,
            lookback_weeks=lookback_weeks,
            points=[
                HistoryPoint(week_start=row["week_start"].date(), value=round(float(row[target_metric.value]), 2))
                for row in history_tail.to_dict(orient="records")
            ],
            summary=(
                f"Showing the most recent {len(history_tail)} clean weekly actuals used by the forecasting pipeline."
            ),
        )

    def forecast_entity(
        self,
        entity_type: EntityType,
        entity_id: str,
        horizon_weeks: int,
        target_metric: TargetMetric,
        price_multiplier: float,
    ) -> ForecastResponse:
        catalog_record = get_entity_catalog_record(self.db, entity_type, entity_id)
        weekly = load_weekly_snapshot(entity_type)

        if catalog_record is not None and catalog_record.status == STATUS_DATA_ISSUE:
            return _zero_issue_forecast(entity_type, entity_id, horizon_weeks, target_metric, catalog_record, weekly)

        assignment = get_assignment(self.db, entity_type, entity_id)
        if assignment is None:
            raise ValueError(f"No cluster mapping found for {entity_type.value} {entity_id}")

        cluster_profile = get_cluster_profile(self.db, entity_type, assignment.cluster_id)
        if cluster_profile is None:
            raise ValueError(f"No cluster profile found for {assignment.cluster_id}")

        model_record = get_model_record(self.db, entity_type, assignment.cluster_id, target_metric.value)
        if model_record is None:
            raise ValueError(
                f"No forecast model found for {entity_type.value} cluster {assignment.cluster_id}. Run training first."
            )

        payload = joblib.load(model_record.artifact_path)
        history = weekly[weekly["entity_id"] == entity_id].sort_values("week_start").copy()
        if history.empty:
            raise ValueError(f"No weekly history found for {entity_type.value} {entity_id}")

        entity_status = catalog_record.status if catalog_record is not None else STATUS_OK
        forecast_strategy = catalog_record.forecast_strategy if catalog_record is not None else STRATEGY_CLUSTER_MODEL
        if entity_status == STATUS_SPARSE_HISTORY:
            forecast_frame = _similar_peer_forecast(
                weekly=weekly,
                history=history,
                payload=payload,
                catalog_record=catalog_record,
                horizon_weeks=horizon_weeks,
                target_metric=target_metric,
                price_multiplier=price_multiplier,
            )
        else:
            forecast_frame = _recursive_forecast(history, payload, horizon_weeks, price_multiplier)

        baseline = float(history[target_metric.value].tail(12).mean()) if not history.empty else 0.0
        first_value = float(forecast_frame.iloc[0]["value"]) if not forecast_frame.empty else 0.0
        change_pct = ((first_value - baseline) / baseline * 100.0) if baseline else None
        total_forecast = float(forecast_frame["value"].sum())
        narrative = _build_narrative(
            entity_type=entity_type,
            entity_id=entity_id,
            cluster_label=cluster_profile.label,
            target_metric=target_metric,
            first_value=first_value,
            baseline=baseline,
            change_pct=change_pct,
            price_multiplier=price_multiplier,
            entity_status=entity_status,
            forecast_strategy=forecast_strategy,
            data_quality_summary=catalog_record.issue_summary if catalog_record is not None else None,
        )
        return ForecastResponse(
            entity_type=entity_type,
            entity_id=entity_id,
            cluster_id=assignment.cluster_id,
            cluster_label=cluster_profile.label,
            entity_status=entity_status,
            forecast_strategy=forecast_strategy,
            target_metric=target_metric,
            horizon_weeks=horizon_weeks,
            baseline_recent_average=round(baseline, 2),
            first_week_change_pct=round(change_pct, 2) if change_pct is not None else None,
            total_forecast=round(total_forecast, 2),
            forecast=[
                ForecastPoint(week_start=row["week_start"].date(), value=round(float(row["value"]), 2))
                for row in forecast_frame.to_dict(orient="records")
            ],
            narrative=narrative,
            data_quality_summary=catalog_record.issue_summary if catalog_record is not None else None,
            data_issue_codes=(catalog_record.issue_codes or []) if catalog_record is not None else [],
        )

    def compare_entities(
        self,
        entity_type: EntityType,
        entity_ids: list[str],
        horizon_weeks: int,
        target_metric: TargetMetric,
        price_multiplier: float,
    ) -> ComparisonResponse:
        forecasts = [
            self.forecast_entity(entity_type, entity_id, horizon_weeks, target_metric, price_multiplier)
            for entity_id in entity_ids
        ]
        ranked = sorted(forecasts, key=lambda item: item.total_forecast, reverse=True)
        leader = ranked[0]
        runner_up = ranked[1]
        delta = leader.total_forecast - runner_up.total_forecast
        summary = (
            f"{leader.entity_id} leads {runner_up.entity_id} by {delta:.2f} "
            f"{target_metric.value} over the next {horizon_weeks} weeks."
        )
        return ComparisonResponse(
            entity_type=entity_type,
            target_metric=target_metric,
            comparison_summary=summary,
            forecasts=forecasts,
        )

    def declining_alerts(
        self,
        entity_type: EntityType,
        target_metric: TargetMetric,
        horizon_weeks: int,
        declining_threshold_pct: float,
        top_k: int,
        cluster_id: str | None = None,
        cluster_label_contains: str | None = None,
    ) -> AlertResponse:
        weekly = load_weekly_snapshot(entity_type)
        profiles = []
        if cluster_id:
            profiles = [get_cluster_profile(self.db, entity_type, cluster_id)]
        elif cluster_label_contains:
            profiles = find_cluster_profiles_by_label(self.db, entity_type, cluster_label_contains)
        else:
            raise ValueError("Provide cluster_id or cluster_label_contains for alerts")

        profiles = [profile for profile in profiles if profile is not None]
        if not profiles:
            raise ValueError("No clusters matched the requested filter")

        alerts: list[AlertItem] = []
        for profile in profiles:
            assignments = list_assignments_for_cluster(self.db, entity_type, profile.cluster_id)
            candidate_ids = [assignment.entity_id for assignment in assignments]
            recent = (
                weekly[weekly["entity_id"].isin(candidate_ids)]
                .groupby("entity_id")[target_metric.value]
                .mean()
                .sort_values(ascending=False)
            )
            for entity_id in recent.index.tolist():
                forecast = self.forecast_entity(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    horizon_weeks=horizon_weeks,
                    target_metric=target_metric,
                    price_multiplier=1.0,
                )
                recent_average = float(
                    weekly[weekly["entity_id"] == entity_id][target_metric.value].tail(12).mean()
                )
                forecast_average = float(np.mean([point.value for point in forecast.forecast]))
                if recent_average <= 0:
                    continue
                decline_pct = max(0.0, (recent_average - forecast_average) / recent_average)
                if decline_pct >= declining_threshold_pct:
                    alerts.append(
                        AlertItem(
                            entity_id=entity_id,
                            cluster_id=forecast.cluster_id,
                            cluster_label=forecast.cluster_label,
                            recent_average=round(recent_average, 2),
                            forecast_average=round(forecast_average, 2),
                            decline_pct=round(decline_pct, 4),
                        )
                    )
                if len(alerts) >= top_k:
                    break
            if len(alerts) >= top_k:
                break

        summary = f"Found {len(alerts)} declining {entity_type.value} forecasts matching the requested cluster filter."
        return AlertResponse(
            entity_type=entity_type,
            target_metric=target_metric,
            alerts=alerts,
            summary=summary,
        )


def _zero_issue_forecast(
    entity_type: EntityType,
    entity_id: str,
    horizon_weeks: int,
    target_metric: TargetMetric,
    catalog_record,
    weekly: pd.DataFrame,
) -> ForecastResponse:
    latest_week = pd.Timestamp(weekly["week_start"].max()) if not weekly.empty else pd.Timestamp.utcnow().normalize()
    forecast_frame = pd.DataFrame(
        {
            "week_start": [latest_week + timedelta(weeks=index + 1) for index in range(horizon_weeks)],
            "value": [0.0] * horizon_weeks,
        }
    )
    cluster_id = catalog_record.cluster_id or f"{entity_type.value}_data_issue"
    cluster_label = catalog_record.cluster_label or "data-quality-holdout"
    summary = catalog_record.issue_summary or "The entity only appears in rows excluded by notebook-style cleaning."
    return ForecastResponse(
        entity_type=entity_type,
        entity_id=entity_id,
        cluster_id=cluster_id,
        cluster_label=cluster_label,
        entity_status=STATUS_DATA_ISSUE,
        forecast_strategy=STRATEGY_ZERO_FORECAST,
        target_metric=target_metric,
        horizon_weeks=horizon_weeks,
        baseline_recent_average=0.0,
        first_week_change_pct=None,
        total_forecast=0.0,
        forecast=[
            ForecastPoint(week_start=row["week_start"].date(), value=0.0)
            for row in forecast_frame.to_dict(orient="records")
        ],
        narrative=f"{entity_type.value.capitalize()} {entity_id} is mapped to a zero forecast because {summary.lower()}",
        data_quality_summary=summary,
        data_issue_codes=catalog_record.issue_codes or [],
    )


def _similar_peer_forecast(
    weekly: pd.DataFrame,
    history: pd.DataFrame,
    payload: dict,
    catalog_record,
    horizon_weeks: int,
    target_metric: TargetMetric,
    price_multiplier: float,
) -> pd.DataFrame:
    peer_ids = catalog_record.nearest_peer_ids or []
    metadata = catalog_record.metadata_json or {}
    weights = metadata.get("peer_weights", [])
    if not peer_ids:
        return _recursive_forecast(history, payload, horizon_weeks, price_multiplier)

    if not weights or len(weights) != len(peer_ids):
        weights = [1.0 / len(peer_ids)] * len(peer_ids)

    target_level = _recent_positive_mean(history[target_metric.value])
    if target_level <= 0:
        target_level = float(metadata.get("recent_average", 0.0)) or float(payload.get("cluster_baseline", 0.0))

    blended = None
    weeks = None
    for peer_id, weight in zip(peer_ids, weights):
        peer_history = weekly[weekly["entity_id"] == peer_id].sort_values("week_start").copy()
        if peer_history.empty:
            continue
        peer_forecast = _recursive_forecast(peer_history, payload, horizon_weeks, price_multiplier)
        peer_level = _recent_positive_mean(peer_history[target_metric.value]) or float(payload.get("cluster_baseline", 1.0))
        scale = target_level / peer_level if peer_level > 0 else 1.0
        values = np.maximum(peer_forecast["value"].to_numpy(dtype=float) * scale, 0.0)
        blended = values * weight if blended is None else blended + values * weight
        weeks = peer_forecast["week_start"]

    if blended is None or weeks is None:
        return _recursive_forecast(history, payload, horizon_weeks, price_multiplier)

    active_weeks = int((history[target_metric.value] > 0).sum())
    if active_weeks >= 2:
        direct = _recursive_forecast(history, payload, horizon_weeks, price_multiplier)
        direct_values = direct["value"].to_numpy(dtype=float)
        blended = 0.65 * blended + 0.35 * direct_values

    return pd.DataFrame({"week_start": list(weeks), "value": np.maximum(blended, 0.0)})


def _recursive_forecast(
    history: pd.DataFrame,
    payload: dict,
    horizon_weeks: int,
    price_multiplier: float,
) -> pd.DataFrame:
    feature_names = payload["feature_names"]
    target_metric = payload["target_metric"]
    lag_weeks = payload["lag_weeks"]
    model = payload["model"]
    state = history.sort_values("week_start").copy()
    output_rows: list[dict] = []

    for _ in range(horizon_weeks):
        next_week = pd.Timestamp(state["week_start"].max()) + timedelta(weeks=1)
        row = _feature_row(state, next_week, target_metric, lag_weeks, price_multiplier)
        value = float(model.predict(pd.DataFrame([row], columns=feature_names))[0])
        value = max(0.0, value)
        if target_metric == "revenue":
            value *= price_multiplier

        next_row = {
            "entity_id": state.iloc[-1]["entity_id"],
            "week_start": next_week,
            "revenue": state.iloc[-1]["revenue"],
            "quantity": state.iloc[-1]["quantity"],
            "invoice_count": state.iloc[-1]["invoice_count"],
            "counterparties": state.iloc[-1]["counterparties"],
            "avg_unit_price": state.iloc[-1]["avg_unit_price"] * price_multiplier,
        }
        next_row[target_metric] = value
        state = pd.concat([state, pd.DataFrame([next_row])], ignore_index=True)
        output_rows.append({"week_start": next_week, "value": value})

    return pd.DataFrame(output_rows)


def _feature_row(
    state: pd.DataFrame,
    next_week: pd.Timestamp,
    target_metric: str,
    lag_weeks: int,
    price_multiplier: float,
) -> dict:
    series = state[target_metric].to_list()
    lags = list(reversed(series[-lag_weeks:]))
    while len(lags) < lag_weeks:
        lags.append(0.0)

    feature_row = {f"{target_metric}_lag_{index + 1}": float(lags[index]) for index in range(lag_weeks)}
    shifted = state[target_metric].tail(8)
    feature_row["rolling_mean_4"] = float(state[target_metric].tail(4).mean()) if not state.empty else 0.0
    feature_row["rolling_mean_8"] = float(shifted.mean()) if not shifted.empty else 0.0
    feature_row["rolling_std_4"] = float(state[target_metric].tail(4).std()) if len(state) >= 2 else 0.0
    feature_row["avg_unit_price_lag_1"] = float(state["avg_unit_price"].iloc[-1] * price_multiplier)
    feature_row["invoice_count_lag_1"] = float(state["invoice_count"].iloc[-1])
    feature_row["counterparties_lag_1"] = float(state["counterparties"].iloc[-1])
    iso_week = int(next_week.isocalendar().week)
    feature_row["week_sin"] = float(np.sin(2 * np.pi * iso_week / 52.0))
    feature_row["week_cos"] = float(np.cos(2 * np.pi * iso_week / 52.0))
    return feature_row


def _recent_positive_mean(values: pd.Series | np.ndarray) -> float:
    series = pd.Series(values, dtype=float)
    positive = series[series > 0]
    if positive.empty:
        return 0.0
    return float(positive.tail(min(4, len(positive))).mean())


def _build_narrative(
    entity_type: EntityType,
    entity_id: str,
    cluster_label: str,
    target_metric: TargetMetric,
    first_value: float,
    baseline: float,
    change_pct: float | None,
    price_multiplier: float,
    entity_status: str,
    forecast_strategy: str,
    data_quality_summary: str | None,
) -> str:
    metric_name = "spend" if entity_type == EntityType.CUSTOMER and target_metric == TargetMetric.REVENUE else target_metric.value
    if change_pct is None:
        delta_text = "with no recent baseline available"
    else:
        direction = "increase" if change_pct >= 0 else "decrease"
        delta_text = f"a {abs(change_pct):.1f}% {direction} versus the recent 12-week average"

    scenario_text = f" under a {price_multiplier:.2f}x price scenario" if price_multiplier != 1.0 else ""

    if entity_status == STATUS_SPARSE_HISTORY:
        prefix = (
            f"{entity_type.value.capitalize()} {entity_id} belongs to the '{cluster_label}' cluster but has sparse history, "
            f"so the forecast is blended from similar entities. The first forecasted week predicts {metric_name} "
            f"of {first_value:.2f}{scenario_text}, implying {delta_text}."
        )
    elif forecast_strategy == STRATEGY_ZERO_FORECAST:
        prefix = (
            f"{entity_type.value.capitalize()} {entity_id} is held out from model-based forecasting and is fixed at 0 "
            f"because the retained source rows did not survive notebook-style cleaning."
        )
    else:
        prefix = (
            f"{entity_type.value.capitalize()} {entity_id} belongs to the '{cluster_label}' cluster. "
            f"The first forecasted week predicts {metric_name} of {first_value:.2f}{scenario_text}, "
            f"which implies {delta_text}."
        )

    if data_quality_summary:
        return f"{prefix} {data_quality_summary}"
    return prefix
