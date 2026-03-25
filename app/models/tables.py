from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class ClusterAssignment(Base):
    __tablename__ = "cluster_assignments"
    __table_args__ = (UniqueConstraint("entity_type", "entity_id", name="uq_assignment_entity"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_type: Mapped[str] = mapped_column(String(32), index=True)
    entity_id: Mapped[str] = mapped_column(String(128), index=True)
    cluster_id: Mapped[str] = mapped_column(String(64), index=True)
    cluster_label: Mapped[str] = mapped_column(String(255))
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ClusterProfile(Base):
    __tablename__ = "cluster_profiles"
    __table_args__ = (UniqueConstraint("entity_type", "cluster_id", name="uq_cluster_profile"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_type: Mapped[str] = mapped_column(String(32), index=True)
    cluster_id: Mapped[str] = mapped_column(String(64), index=True)
    label: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text)
    member_count: Mapped[int] = mapped_column(Integer)
    prototype_series: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    summary_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ForecastModelRecord(Base):
    __tablename__ = "forecast_model_records"
    __table_args__ = (
        UniqueConstraint(
            "entity_type",
            "cluster_id",
            "target_metric",
            name="uq_forecast_model_record",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_type: Mapped[str] = mapped_column(String(32), index=True)
    cluster_id: Mapped[str] = mapped_column(String(64), index=True)
    target_metric: Mapped[str] = mapped_column(String(32), index=True)
    model_name: Mapped[str] = mapped_column(String(128))
    artifact_path: Mapped[str] = mapped_column(String(512))
    metric_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    feature_names: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    metrics_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EntityCatalogRecord(Base):
    __tablename__ = "entity_catalog_records"
    __table_args__ = (UniqueConstraint("entity_type", "entity_id", name="uq_entity_catalog_record"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_type: Mapped[str] = mapped_column(String(32), index=True)
    entity_id: Mapped[str] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    forecast_strategy: Mapped[str] = mapped_column(String(64))
    issue_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    issue_codes: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    valid_transaction_count: Mapped[int] = mapped_column(Integer, default=0)
    issue_transaction_count: Mapped[int] = mapped_column(Integer, default=0)
    active_weeks: Mapped[int] = mapped_column(Integer, default=0)
    total_weeks: Mapped[int] = mapped_column(Integer, default=0)
    cluster_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    cluster_label: Mapped[str | None] = mapped_column(String(255), nullable=True)
    nearest_peer_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
