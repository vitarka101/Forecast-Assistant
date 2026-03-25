from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import settings
from app.repositories.catalog import replace_assignments, replace_cluster_profiles, replace_entity_catalog, upsert_model_record
from app.schemas.common import EntityType, TargetMetric
from app.schemas.requests import TrainPipelineRequest
from app.schemas.responses import EntityTrainingSummary, TrainPipelineResponse
from app.services.clustering import cluster_entities
from app.services.data_loader import load_transactions
from app.services.entity_catalog import STATUS_DATA_ISSUE, STATUS_SPARSE_HISTORY, build_entity_catalog
from app.services.features import build_entity_weekly_metrics
from app.services.forecasting import save_weekly_snapshot, train_cluster_model


class TrainingPipelineService:
    def __init__(self, db: Session):
        self.db = db

    def run(self, request: TrainPipelineRequest) -> TrainPipelineResponse:
        data_path = request.data_path or settings.raw_data_path
        transactions = load_transactions(data_path)
        summaries: list[EntityTrainingSummary] = []

        config = [
            (EntityType.PRODUCT, request.product_clusters),
            (EntityType.CUSTOMER, request.customer_clusters),
        ]

        for entity_type, cluster_count in config:
            weekly = build_entity_weekly_metrics(transactions, entity_type)
            save_weekly_snapshot(entity_type, weekly)

            result = cluster_entities(
                weekly=weekly,
                entity_type=entity_type,
                target_column=settings.default_target_metric,
                n_clusters=cluster_count,
                min_history_weeks=request.min_history_weeks,
                max_entities_for_dtw=request.max_entities_for_dtw,
            )

            replace_assignments(
                self.db,
                entity_type,
                [
                    {
                        "entity_id": row["entity_id"],
                        "cluster_id": row["cluster_id"],
                        "cluster_label": row["label"],
                        "metadata_json": {"cluster_index": int(row["cluster_index"])},
                    }
                    for row in result.assignments.to_dict(orient="records")
                ],
            )
            replace_cluster_profiles(
                self.db,
                entity_type,
                [
                    {
                        "cluster_id": profile["cluster_id"],
                        "label": profile["label"],
                        "description": profile["description"],
                        "member_count": int(profile["member_count"]),
                        "prototype_series": profile["prototype_series"],
                        "summary_json": profile["summary_json"],
                    }
                    for profile in result.profiles
                ],
            )

            entity_catalog = build_entity_catalog(
                transactions=transactions,
                weekly=weekly,
                assignments=result.assignments,
                entity_type=entity_type,
                min_history_weeks=request.min_history_weeks,
                lag_weeks=request.lag_weeks,
                target_column=settings.default_target_metric,
            )
            replace_entity_catalog(
                self.db,
                entity_type,
                entity_catalog.to_dict(orient="records"),
            )

            trained = 0
            for cluster_id in sorted(result.assignments["cluster_id"].unique().tolist()):
                training_result = train_cluster_model(
                    weekly=weekly,
                    assignments=result.assignments,
                    entity_type=entity_type,
                    cluster_id=cluster_id,
                    target_metric=TargetMetric(settings.default_target_metric),
                    lag_weeks=request.lag_weeks,
                )
                if training_result is None:
                    continue
                upsert_model_record(
                    self.db,
                    {
                        "entity_type": entity_type.value,
                        "cluster_id": cluster_id,
                        "target_metric": settings.default_target_metric,
                        "model_name": training_result.model_name,
                        "artifact_path": training_result.artifact_path,
                        "metric_name": "mae",
                        "metric_value": training_result.metrics.get("mae"),
                        "feature_names": training_result.feature_names,
                        "metrics_json": training_result.metrics,
                    },
                )
                trained += 1

            summaries.append(
                EntityTrainingSummary(
                    entity_type=entity_type,
                    weekly_rows=len(weekly),
                    mapped_entities=result.assignments["entity_id"].nunique(),
                    clusters_built=result.selected_cluster_count,
                    models_trained=trained,
                    issue_only_entities=int((entity_catalog["status"] == STATUS_DATA_ISSUE).sum()),
                    sparse_history_entities=int((entity_catalog["status"] == STATUS_SPARSE_HISTORY).sum()),
                    largest_cluster_share=round(result.max_cluster_share, 4),
                )
            )

        return TrainPipelineResponse(
            data_path=str(Path(data_path).resolve()),
            transactions_loaded=len(transactions),
            entities=summaries,
            message="Retail router-executor pipeline completed successfully.",
        )
