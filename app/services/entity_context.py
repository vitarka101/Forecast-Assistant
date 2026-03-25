from functools import lru_cache

import pandas as pd

from app.core.config import settings
from app.schemas.common import EntityType
from app.schemas.responses import EntityContextResponse
from app.services.data_loader import load_transactions


@lru_cache(maxsize=1)
def _metadata_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    transactions = load_transactions(settings.raw_data_path)

    products = transactions[transactions["stock_code"].notna()].copy()
    products["description_clean"] = products["description"].astype(str).str.strip()
    products = products[~products["description_clean"].str.lower().isin(["", "nan", "none"])]
    product_lookup = (
        products.groupby(["stock_code", "description_clean"], as_index=False)
        .size()
        .sort_values(["stock_code", "size", "description_clean"], ascending=[True, False, True])
        .drop_duplicates(subset=["stock_code"])
        .rename(columns={"stock_code": "entity_id", "description_clean": "short_label"})
    )[["entity_id", "short_label"]]

    customers = transactions[transactions["customer_id"].notna()].copy()
    customer_lookup = (
        customers.groupby("customer_id", as_index=False)
        .agg(
            primary_country=("country", lambda values: values.mode().iat[0] if not values.mode().empty else ""),
            distinct_products=("stock_code", lambda values: int(values.dropna().nunique())),
        )
        .rename(columns={"customer_id": "entity_id"})
    )
    customer_lookup["note"] = customer_lookup.apply(
        lambda row: (
            f"{row['primary_country']} customer · {row['distinct_products']} distinct products"
            if row["primary_country"]
            else f"Retail customer · {row['distinct_products']} distinct products"
        ),
        axis=1,
    )
    return product_lookup, customer_lookup[["entity_id", "note"]]


def get_entity_context(entity_type: EntityType, entity_id: str) -> EntityContextResponse:
    product_lookup, customer_lookup = _metadata_frames()

    if entity_type == EntityType.PRODUCT:
        match = product_lookup[product_lookup["entity_id"] == entity_id]
        if match.empty:
            return EntityContextResponse(entity_type=entity_type, entity_id=entity_id)
        return EntityContextResponse(
            entity_type=entity_type,
            entity_id=entity_id,
            short_label=str(match.iloc[0]["short_label"]),
        )

    match = customer_lookup[customer_lookup["entity_id"] == entity_id]
    if match.empty:
        return EntityContextResponse(entity_type=entity_type, entity_id=entity_id)
    return EntityContextResponse(
        entity_type=entity_type,
        entity_id=entity_id,
        note=str(match.iloc[0]["note"]),
    )
