from collections.abc import Iterable

from sqlalchemy import delete, insert, select
from sqlalchemy.orm import Session

from app.models.tables import ClusterAssignment, ClusterProfile, EntityCatalogRecord, ForecastModelRecord
from app.schemas.common import EntityType


def replace_assignments(db: Session, entity_type: EntityType, rows: Iterable[dict]) -> None:
    db.execute(delete(ClusterAssignment).where(ClusterAssignment.entity_type == entity_type.value))
    db.add_all([ClusterAssignment(entity_type=entity_type.value, **row) for row in rows])
    db.commit()


def replace_cluster_profiles(db: Session, entity_type: EntityType, rows: Iterable[dict]) -> None:
    db.execute(delete(ClusterProfile).where(ClusterProfile.entity_type == entity_type.value))
    db.add_all([ClusterProfile(entity_type=entity_type.value, **row) for row in rows])
    db.commit()


def upsert_model_record(db: Session, payload: dict) -> None:
    stmt = select(ForecastModelRecord).where(
        ForecastModelRecord.entity_type == payload["entity_type"],
        ForecastModelRecord.cluster_id == payload["cluster_id"],
        ForecastModelRecord.target_metric == payload["target_metric"],
    )
    record = db.execute(stmt).scalar_one_or_none()
    if record is None:
        record = ForecastModelRecord(**payload)
        db.add(record)
    else:
        for key, value in payload.items():
            setattr(record, key, value)
    db.commit()


def replace_entity_catalog(db: Session, entity_type: EntityType, rows: Iterable[dict]) -> None:
    db.execute(delete(EntityCatalogRecord).where(EntityCatalogRecord.entity_type == entity_type.value))
    for row in rows:
        db.execute(insert(EntityCatalogRecord).values(entity_type=entity_type.value, **row))
    db.commit()


def get_assignment(db: Session, entity_type: EntityType, entity_id: str) -> ClusterAssignment | None:
    stmt = select(ClusterAssignment).where(
        ClusterAssignment.entity_type == entity_type.value,
        ClusterAssignment.entity_id == entity_id,
    )
    return db.execute(stmt).scalar_one_or_none()


def get_entity_catalog_record(db: Session, entity_type: EntityType, entity_id: str) -> EntityCatalogRecord | None:
    stmt = select(EntityCatalogRecord).where(
        EntityCatalogRecord.entity_type == entity_type.value,
        EntityCatalogRecord.entity_id == entity_id,
    )
    return db.execute(stmt).scalar_one_or_none()


def get_cluster_profile(db: Session, entity_type: EntityType, cluster_id: str) -> ClusterProfile | None:
    stmt = select(ClusterProfile).where(
        ClusterProfile.entity_type == entity_type.value,
        ClusterProfile.cluster_id == cluster_id,
    )
    return db.execute(stmt).scalar_one_or_none()


def find_cluster_profiles_by_label(
    db: Session,
    entity_type: EntityType,
    label_fragment: str,
) -> list[ClusterProfile]:
    stmt = select(ClusterProfile).where(
        ClusterProfile.entity_type == entity_type.value,
        ClusterProfile.label.ilike(f"%{label_fragment}%"),
    )
    return list(db.execute(stmt).scalars().all())


def get_model_record(
    db: Session,
    entity_type: EntityType,
    cluster_id: str,
    target_metric: str,
) -> ForecastModelRecord | None:
    stmt = select(ForecastModelRecord).where(
        ForecastModelRecord.entity_type == entity_type.value,
        ForecastModelRecord.cluster_id == cluster_id,
        ForecastModelRecord.target_metric == target_metric,
    )
    return db.execute(stmt).scalar_one_or_none()


def list_assignments_for_cluster(
    db: Session,
    entity_type: EntityType,
    cluster_id: str,
) -> list[ClusterAssignment]:
    stmt = select(ClusterAssignment).where(
        ClusterAssignment.entity_type == entity_type.value,
        ClusterAssignment.cluster_id == cluster_id,
    )
    return list(db.execute(stmt).scalars().all())
