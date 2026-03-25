from pathlib import Path

import numpy as np
import pandas as pd


COLUMN_MAP = {
    "Invoice": "invoice_no",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "unit_price",
    "Customer ID": "customer_id",
    "Country": "country",
}

ISSUE_FLAG_COLUMNS = {
    "duplicate_row": "is_duplicate",
    "cancelled_invoice": "is_cancelled_invoice",
    "non_positive_quantity": "has_non_positive_quantity",
    "non_positive_price": "has_non_positive_price",
    "missing_invoice_date": "missing_invoice_date",
    "missing_stock_code": "missing_stock_code",
    "missing_customer_id": "missing_customer_id",
}


def _normalize_identifier(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    return text.upper()


def load_transactions(data_path: str | Path) -> pd.DataFrame:
    workbook = pd.read_excel(data_path, sheet_name=None, engine="openpyxl")
    frames: list[pd.DataFrame] = []
    for sheet_name, frame in workbook.items():
        normalized = frame.rename(columns=COLUMN_MAP).copy()
        normalized["source_sheet"] = sheet_name
        frames.append(normalized)

    data = pd.concat(frames, ignore_index=True)
    required = [
        "invoice_no",
        "stock_code",
        "quantity",
        "invoice_date",
        "unit_price",
        "customer_id",
        "country",
    ]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    data["invoice_no"] = data["invoice_no"].astype(str).str.strip()
    data["stock_code"] = data["stock_code"].apply(_normalize_identifier)
    data["customer_id"] = data["customer_id"].apply(_normalize_identifier)
    if "description" in data.columns:
        data["description"] = data["description"].astype(str).str.strip()
    else:
        data["description"] = ""
    data["country"] = data["country"].astype(str).str.strip()
    data["quantity"] = pd.to_numeric(data["quantity"], errors="coerce")
    data["unit_price"] = pd.to_numeric(data["unit_price"], errors="coerce")
    data["invoice_date"] = pd.to_datetime(data["invoice_date"], errors="coerce")

    data["is_duplicate"] = data.duplicated(keep="first")
    data["is_cancelled_invoice"] = data["invoice_no"].str.startswith("C", na=False)
    data["has_non_positive_quantity"] = data["quantity"].fillna(0).le(0)
    data["has_non_positive_price"] = data["unit_price"].fillna(0).le(0)
    data["missing_invoice_date"] = data["invoice_date"].isna()
    data["missing_stock_code"] = data["stock_code"].isna()
    data["missing_customer_id"] = data["customer_id"].isna()

    flag_arrays = {code: data[flag_column].to_numpy(dtype=bool) for code, flag_column in ISSUE_FLAG_COLUMNS.items()}
    data["row_issue_codes"] = [
        [code for code, flags in flag_arrays.items() if flags[index]]
        for index in range(len(data))
    ]
    data["row_issue_count"] = data["row_issue_codes"].str.len()

    product_blockers = [
        "is_duplicate",
        "is_cancelled_invoice",
        "has_non_positive_quantity",
        "has_non_positive_price",
        "missing_invoice_date",
        "missing_stock_code",
    ]
    customer_blockers = [*product_blockers, "missing_customer_id"]
    data["valid_for_product"] = ~data[product_blockers].any(axis=1)
    data["valid_for_customer"] = ~data[customer_blockers].any(axis=1)

    quantity = data["quantity"].fillna(0.0)
    unit_price = data["unit_price"].fillna(0.0)
    data["revenue"] = np.where(data["valid_for_product"], quantity * unit_price, 0.0)
    data["week_start"] = data["invoice_date"].dt.to_period("W").dt.start_time
    return data
