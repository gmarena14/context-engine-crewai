from __future__ import annotations

import ast
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


def build_item_360(items: pd.DataFrame) -> Tuple[pd.DataFrame, "HealthReport"]:
    """
    Build a compact 360 profile for each item_id, plus a basic health report.
    Expected columns in items:
      - item_id, title, seller_id, price, available_quantity, sold_quantity, tags, attributes
    """
    df = items.copy()

    # ---- basic type hygiene ----
    df["item_id"] = df["item_id"].astype(str)
    df["seller_id"] = df["seller_id"].astype(str)
    df["title"] = df["title"].astype(str)

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["available_quantity"] = pd.to_numeric(df["available_quantity"], errors="coerce")
    df["sold_quantity"] = pd.to_numeric(df["sold_quantity"], errors="coerce")

    # ----------------------------
    # tags: normalize to list[str]
    # ----------------------------
    def normalize_tags(x):
        # None
        if x is None:
            return []

        # Scalar NaN (solo si es escalar)
        if np.isscalar(x) and pd.isna(x):
            return []

        # list/tuple/set/np.ndarray/pd.Series
        if isinstance(x, (list, tuple, set, np.ndarray, pd.Series)):
            out = []
            for t in list(x):
                if t is None:
                    continue
                if np.isscalar(t) and pd.isna(t):
                    continue
                s = str(t).strip().lower()
                if s:
                    out.append(s)
            return out

        # string (a veces viene como "['work','gaming']")
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return []
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = ast.literal_eval(s)
                    return normalize_tags(parsed)
                except Exception:
                    # si no se puede parsear, lo tratamos como tag único
                    return [s.lower()]
            return [s.lower()]

        # fallback: cualquier otro objeto
        s = str(x).strip().lower()
        return [s] if s else []

    df["tags_norm"] = df["tags"].apply(normalize_tags)

    # -----------------------------------
    # attributes: normalize to dict safely
    # -----------------------------------
    def normalize_attributes(a):
        if a is None:
            return {}
        if isinstance(a, dict):
            return a
        # scalar NaN
        if np.isscalar(a) and pd.isna(a):
            return {}
        # string JSON-ish
        if isinstance(a, str):
            s = a.strip()
            if not s:
                return {}
            try:
                parsed = ast.literal_eval(s)  # funciona para dicts estilo Python
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    df["attributes_norm"] = df["attributes"].apply(normalize_attributes)

    # attributes: extract common fields
    df["category"] = df["attributes_norm"].apply(lambda d: _safe_get(d, "category"))
    df["brand"] = df["attributes_norm"].apply(lambda d: _safe_get(d, "brand"))
    df["model"] = df["attributes_norm"].apply(lambda d: _safe_get(d, "model"))

    # title signals
    df["title_norm"] = df["title"].astype(str).str.lower().str.strip()
    df["title_len"] = df["title_norm"].str.len()

    # inventory / demand signals (evita división por 0)
    df["stock_ratio"] = (df["available_quantity"].fillna(0) + 1) / (df["sold_quantity"].fillna(0) + 1)
    denom = (df["available_quantity"].fillna(0) + df["sold_quantity"].fillna(0) + 1e-9)
    df["sell_through"] = df["sold_quantity"].fillna(0) / denom

    # derived
    df["price_bucket"] = df["price"].apply(lambda p: _bucket_price(p) if pd.notna(p) else None)
    df["n_tags"] = df["tags_norm"].apply(len)

    # ---- item_360 output (1 row per item_id) ----
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

    if df["price"].notna().any():
        price_vals = df["price"].to_numpy(dtype=float)
        price_summary = {
            "min": float(np.nanmin(price_vals)),
            "p50": float(np.nanmedian(price_vals)),
            "p95": float(np.nanpercentile(price_vals, 95)),
            "max": float(np.nanmax(price_vals)),
        }
    else:
        price_summary = {"min": float("nan"), "p50": float("nan"), "p95": float("nan"), "max": float("nan")}

    category_counts = df["category"].fillna("unknown").value_counts().head(30).to_dict()

    report = HealthReport(
        n_rows=n_rows,
        n_unique_item_id=n_unique,
        duplicate_item_id_rows=dup_rows,
        missing_pct_by_col=missing_pct,
        price_summary=price_summary,
        category_counts=category_counts,
    )

    return item_360, report
