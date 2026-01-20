from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd


@dataclass
class HealthReport:
    n_rows: int
    n_unique_item_id: int
    duplicate_item_id_rows: int
    missing_pct_by_col: Dict[str, float]
    price_summary: Dict[str, float]
    category_counts: Dict[str, int]


def _safe_get(d: Any, key: str, default=None):
    if isinstance(d, dict):
        return d.get(key, default)
    return default


def _bucket_price(price: float) -> str:
    # buckets sencillos para demo
    if price < 500:
        return "<500"
    if price < 1000:
        return "500-999"
    if price < 2000:
        return "1000-1999"
    if price < 3000:
        return "2000-2999"
    if price < 5000:
        return "3000-4999"
    return "5000+"


def build_item_360(items: pd.DataFrame) -> Tuple[pd.DataFrame, HealthReport]:
    """
    Build a compact 360 profile for each item_id, plus a basic health report.
    Expected columns in items:
      - item_id, title, seller_id, price, available_quantity, sold_quantity, tags, attributes
    """
    df = items.copy()

    # ---- basic type hygiene ----
    df["item_id"] = df["item_id"].astype(str)
    df["seller_id"] = df["seller_id"].astype(str)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["available_quantity"] = pd.to_numeric(df["available_quantity"], errors="coerce")
    df["sold_quantity"] = pd.to_numeric(df["sold_quantity"], errors="coerce")

    # tags: ensure list
    def normalize_tags(x):
        if isinstance(x, list):
            return [str(t).strip().lower() for t in x if str(t).strip()]
        if pd.isna(x):
            return []
        return [str(x).strip().lower()]

    df["tags_norm"] = df["tags"].apply(normalize_tags)

    # attributes: extract common fields
    df["category"] = df["attributes"].apply(lambda d: _safe_get(d, "category"))
    df["brand"] = df["attributes"].apply(lambda d: _safe_get(d, "brand"))
    df["model"] = df["attributes"].apply(lambda d: _safe_get(d, "model"))

    # title signals
    df["title_norm"] = df["title"].astype(str).str.lower().str.strip()
    df["title_len"] = df["title_norm"].str.len()

    # inventory / demand signals
    df["stock_ratio"] = (df["available_quantity"] + 1) / (df["sold_quantity"] + 1)
    df["sell_through"] = df["sold_quantity"] / (df["available_quantity"] + df["sold_quantity"] + 1e-9)

    # derived
    df["price_bucket"] = df["price"].apply(lambda p: _bucket_price(p) if pd.notna(p) else None)
    df["n_tags"] = df["tags_norm"].apply(len)

    # ---- item_360 output (1 row per item_id) ----
    # En este dataset es 1 row por item_id, pero lo dejamos robusto:
    agg = {
        "title": "first",
        "seller_id": "first",
        "price": "first",
        "price_bucket": "first",
        "available_quantity": "first",
        "sold_quantity": "first",
        "stock_ratio": "first",
        "sell_through": "first",
        "category": "first",
        "brand": "first",
        "model": "first",
        "n_tags": "first",
        "tags_norm": "first",
        "title_len": "first",
    }
    item_360 = df.groupby("item_id", as_index=False).agg(agg)

    # ---- health report ----
    n_rows = len(df)
    n_unique = df["item_id"].nunique()
    dup_rows = int(n_rows - n_unique)

    missing_pct = (df.isna().mean() * 100).round(2).to_dict()

    price_summary = {
        "min": float(np.nanmin(df["price"].values)) if df["price"].notna().any() else float("nan"),
        "p50": float(np.nanmedian(df["price"].values)) if df["price"].notna().any() else float("nan"),
        "p95": float(np.nanpercentile(df["price"].values, 95)) if df["price"].notna().any() else float("nan"),
        "max": float(np.nanmax(df["price"].values)) if df["price"].notna().any() else float("nan"),
    }

    category_counts = (
        df["category"].fillna("unknown").value_counts().head(30).to_dict()
    )

    report = HealthReport(
        n_rows=n_rows,
        n_unique_item_id=n_unique,
        duplicate_item_id_rows=dup_rows,
        missing_pct_by_col=missing_pct,
        price_summary=price_summary,
        category_counts=category_counts,
    )

    return item_360, report
