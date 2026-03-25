from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from app.schemas.common import EntityType
from app.services.features import build_shape_matrix, normalize_shape_matrix


@dataclass
class ClusteringResult:
    assignments: pd.DataFrame
    profiles: list[dict]
    selected_cluster_count: int
    max_cluster_share: float


def dtw_distance(series_a: np.ndarray, series_b: np.ndarray) -> float:
    n = len(series_a)
    m = len(series_b)
    matrix = np.full((n + 1, m + 1), np.inf)
    matrix[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(series_a[i - 1] - series_b[j - 1])
            matrix[i, j] = cost + min(matrix[i - 1, j], matrix[i, j - 1], matrix[i - 1, j - 1])
    return float(matrix[n, m])


def cluster_entities(
    weekly: pd.DataFrame,
    entity_type: EntityType,
    target_column: str,
    n_clusters: int,
    min_history_weeks: int,
    max_entities_for_dtw: int,
) -> ClusteringResult:
    shape_matrix = build_shape_matrix(weekly, target_column)
    normalized = normalize_shape_matrix(shape_matrix)
    active_weeks = (shape_matrix > 0).sum(axis=1)
    candidate_ids = active_weeks[active_weeks >= min_history_weeks].index.tolist()
    if not candidate_ids:
        candidate_ids = shape_matrix.index.tolist()
    if not candidate_ids:
        raise ValueError(f"No entities available for {entity_type.value} clustering")

    selected_cluster_count = _recommended_cluster_count(n_clusters, len(candidate_ids), entity_type)
    training_ids = _select_training_ids(shape_matrix, normalized, candidate_ids, max_entities_for_dtw)
    training_matrix = normalized.loc[training_ids]

    if training_matrix.empty:
        raise ValueError(f"No entities available for {entity_type.value} clustering")

    cluster_count = max(1, min(selected_cluster_count, len(training_matrix)))
    if cluster_count == 1:
        labels = np.zeros(len(training_matrix), dtype=int)
    else:
        distance_matrix = _pairwise_dtw(training_matrix.to_numpy(dtype=float))
        model = AgglomerativeClustering(n_clusters=cluster_count, metric="precomputed", linkage="average")
        labels = model.fit_predict(distance_matrix)

    labeled = pd.DataFrame({"entity_id": training_ids, "cluster_index": labels})
    prototypes = {
        cluster_index: training_matrix.loc[labeled[labeled["cluster_index"] == cluster_index]["entity_id"]]
        .mean(axis=0)
        .to_numpy(dtype=float)
        for cluster_index in sorted(labeled["cluster_index"].unique())
    }

    assignments = _balanced_assignments(normalized, prototypes, entity_type)
    profiles = _build_profiles(assignments, normalized, shape_matrix, entity_type)
    assignments = assignments.merge(
        pd.DataFrame(profiles)[["cluster_id", "label", "description"]],
        on="cluster_id",
        how="left",
    )
    cluster_sizes = assignments.groupby("cluster_id")["entity_id"].nunique()
    max_cluster_share = float(cluster_sizes.max() / max(cluster_sizes.sum(), 1)) if not cluster_sizes.empty else 0.0
    return ClusteringResult(
        assignments=assignments,
        profiles=profiles,
        selected_cluster_count=len(profiles),
        max_cluster_share=max_cluster_share,
    )


def _recommended_cluster_count(requested_clusters: int, eligible_entities: int, entity_type: EntityType) -> int:
    if eligible_entities <= 1:
        return 1

    target_entities_per_cluster = 40
    derived = int(round(math.sqrt(eligible_entities / target_entities_per_cluster)))
    if entity_type == EntityType.PRODUCT and eligible_entities >= 3000:
        derived = max(derived, 8)
    if entity_type == EntityType.CUSTOMER and eligible_entities >= 1000:
        derived = max(derived, 5)

    minimum = 2
    upper_bound = min(eligible_entities, max(2, eligible_entities // 30), 18 if entity_type == EntityType.PRODUCT else 12)
    baseline = max(minimum, requested_clusters, derived)
    return min(eligible_entities, max(minimum, min(baseline, upper_bound)))


def _select_training_ids(
    shape_matrix: pd.DataFrame,
    normalized: pd.DataFrame,
    candidate_ids: list[str],
    max_entities_for_dtw: int,
) -> list[str]:
    if len(candidate_ids) <= max_entities_for_dtw:
        return candidate_ids

    summary = pd.DataFrame(
        {
            "entity_id": candidate_ids,
            "active_weeks": (shape_matrix.loc[candidate_ids] > 0).sum(axis=1).to_numpy(dtype=int),
            "total_signal": shape_matrix.loc[candidate_ids].sum(axis=1).to_numpy(dtype=float),
            "volatility": shape_matrix.loc[candidate_ids].std(axis=1).to_numpy(dtype=float),
        }
    )
    summary["active_bin"] = _quantile_bins(summary["active_weeks"], 5)
    summary["signal_bin"] = _quantile_bins(np.log1p(summary["total_signal"]), 5)
    summary["volatility_bin"] = _quantile_bins(summary["volatility"], 4)
    summary["stratum"] = (
        summary["active_bin"].astype(str)
        + "|"
        + summary["signal_bin"].astype(str)
        + "|"
        + summary["volatility_bin"].astype(str)
    )

    selected: list[str] = []
    for _, group in summary.groupby("stratum"):
        ordered = group.sort_values(["active_weeks", "total_signal"], ascending=[False, False])
        take = max(1, math.ceil(max_entities_for_dtw / max(summary["stratum"].nunique(), 1)))
        selected.extend(ordered.head(take)["entity_id"].tolist())

    if len(selected) < max_entities_for_dtw:
        remaining = summary[~summary["entity_id"].isin(selected)].sort_values(
            ["active_weeks", "total_signal", "volatility"], ascending=[False, False, False]
        )
        selected.extend(remaining.head(max_entities_for_dtw - len(selected))["entity_id"].tolist())

    deduped = list(dict.fromkeys(selected))
    if len(deduped) > max_entities_for_dtw:
        scores = normalized.loc[deduped].abs().sum(axis=1).sort_values(ascending=False)
        deduped = scores.head(max_entities_for_dtw).index.tolist()
    return deduped


def _quantile_bins(values: pd.Series | np.ndarray, q: int) -> pd.Series:
    series = pd.Series(values).rank(method="first")
    bins = min(q, max(1, series.nunique()))
    return pd.qcut(series, q=bins, labels=False, duplicates="drop")


def _pairwise_dtw(values: np.ndarray) -> np.ndarray:
    size = values.shape[0]
    matrix = np.zeros((size, size), dtype=float)
    for i in range(size):
        for j in range(i + 1, size):
            distance = dtw_distance(values[i], values[j])
            matrix[i, j] = distance
            matrix[j, i] = distance
    return matrix


def _balanced_assignments(
    normalized: pd.DataFrame,
    prototypes: dict[int, np.ndarray],
    entity_type: EntityType,
) -> pd.DataFrame:
    cluster_ids = sorted(prototypes)
    if not cluster_ids:
        raise ValueError(f"No cluster prototypes were generated for {entity_type.value}")

    distance_rows: list[dict] = []
    for entity_id, row in normalized.iterrows():
        distances = {cluster_index: dtw_distance(row.to_numpy(dtype=float), prototypes[cluster_index]) for cluster_index in cluster_ids}
        ordered = sorted(distances.items(), key=lambda item: item[1])
        best = ordered[0][1]
        second = ordered[1][1] if len(ordered) > 1 else ordered[0][1]
        distance_rows.append(
            {
                "entity_id": entity_id,
                "ordered_clusters": [cluster_index for cluster_index, _ in ordered],
                "distances": distances,
                "margin": float(second - best),
            }
        )

    total_entities = len(distance_rows)
    average_size = max(1, math.ceil(total_entities / len(cluster_ids)))
    max_capacity = max(average_size, math.ceil(average_size * 1.6))
    counts = {cluster_index: 0 for cluster_index in cluster_ids}
    assignments: list[dict] = []

    for row in sorted(distance_rows, key=lambda item: item["margin"], reverse=True):
        selected_cluster = row["ordered_clusters"][0]
        for cluster_index in row["ordered_clusters"]:
            if counts[cluster_index] < max_capacity:
                selected_cluster = cluster_index
                break
        counts[selected_cluster] += 1
        assignments.append(
            {
                "entity_id": row["entity_id"],
                "cluster_id": f"{entity_type.value}_cluster_{selected_cluster + 1:02d}",
                "cluster_index": int(selected_cluster),
            }
        )

    return pd.DataFrame(assignments)


def _build_profiles(
    assignments: pd.DataFrame,
    normalized: pd.DataFrame,
    original: pd.DataFrame,
    entity_type: EntityType,
) -> list[dict]:
    profiles: list[dict] = []
    noun = "products" if entity_type == EntityType.PRODUCT else "customers"

    for cluster_id, members in assignments.groupby("cluster_id"):
        member_ids = members["entity_id"].tolist()
        prototype = normalized.loc[member_ids].mean(axis=0).to_numpy(dtype=float)
        raw_prototype = original.loc[member_ids].mean(axis=0).to_numpy(dtype=float)
        slope = np.polyfit(np.arange(len(prototype)), prototype, 1)[0] if len(prototype) > 1 else 0.0
        volatility = float(np.std(prototype))

        if slope > 0.08:
            trend = "growth"
        elif slope < -0.08:
            trend = "declining"
        else:
            trend = "flat"

        stability = "high-volatility" if volatility > 0.9 else "steady"
        label = f"{stability}-{trend}-{noun}"
        description = (
            f"{cluster_id} groups {noun} with {trend} time-series shape and "
            f"{'higher' if stability == 'high-volatility' else 'lower'} weekly volatility."
        )
        profiles.append(
            {
                "cluster_id": cluster_id,
                "label": label,
                "description": description,
                "member_count": len(member_ids),
                "prototype_series": [float(value) for value in raw_prototype],
                "summary_json": {
                    "trend_slope": float(slope),
                    "volatility": volatility,
                    "member_ids_sample": member_ids[:10],
                },
            }
        )
    return profiles
