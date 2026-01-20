# src/data_simulator.py
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SimConfig:
    n_items: int = 100_000
    n_sellers: int = 5_000
    n_events: int = 10_000
    seed: int = 42


CATEGORIES = {
    "laptop": {
        "brands": ["Lenovo", "HP", "Dell", "ASUS", "Acer", "MSI"],
        "models": ["IdeaPad", "Pavilion", "Inspiron", "VivoBook", "Nitro", "Katana"],
        "tags": ["ssd", "ram", "gaming", "ultrabook", "work", "student"],
        "price_range": (1400, 6500),
    },
    "tv": {
        "brands": ["Samsung", "LG", "Sony", "TCL", "Hisense"],
        "models": ["Crystal", "NanoCell", "Bravia", "QLED", "UHD"],
        "tags": ["4k", "smart", "hdr", "55pulgadas", "65pulgadas"],
        "price_range": (1200, 9000),
    },
    "headphones": {
        "brands": ["Sony", "Bose", "JBL", "Sennheiser", "Apple"],
        "models": ["WH", "QuietComfort", "Tune", "HD", "AirPods"],
        "tags": ["bluetooth", "cancelacion_ruido", "microfono", "deporte"],
        "price_range": (150, 2500),
    },
    "phone": {
        "brands": ["Samsung", "Apple", "Xiaomi", "Motorola", "Google"],
        "models": ["Galaxy", "iPhone", "Redmi", "Edge", "Pixel"],
        "tags": ["5g", "oled", "camara_pro", "bateria_larga", "dual_sim"],
        "price_range": (400, 8000),
    },
    "camera": {
        "brands": ["Canon", "Nikon", "Sony", "Fujifilm", "GoPro"],
        "models": ["EOS", "D", "Alpha", "X", "Hero"],
        "tags": ["4k", "estabilizacion", "lente_kit", "vlog"],
        "price_range": (600, 12000),
    },
}

CONDITIONS = ["new", "used", "refurbished"]


def _rng(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def _sample_price(low: float, high: float) -> float:
    # lognormal-ish dentro de rango
    mu = np.log((low + high) / 2)
    sigma = 0.55
    x = np.random.lognormal(mean=mu, sigma=sigma)
    return float(np.clip(x, low, high))


def _make_attributes(cat: str, brand: str, model: str) -> dict[str, Any]:
    # dict sencillo pero realista
    attrs: dict[str, Any] = {
        "category": cat,
        "brand": brand,
        "model": model,
        "condition": random.choice(CONDITIONS),
    }

    if cat == "laptop":
        attrs["ram"] = random.choice([8, 16, 32, 64])
        attrs["storage"] = random.choice([256, 512, 1024, 2048])
        attrs["screen_size"] = random.choice([13.3, 14.0, 15.6, 16.0, 17.3])
        attrs["cpu"] = random.choice(["i5", "i7", "i9", "ryzen5", "ryzen7"])
    elif cat == "tv":
        attrs["screen_size"] = random.choice([43, 50, 55, 65, 75])
        attrs["panel"] = random.choice(["led", "qled", "oled"])
        attrs["resolution"] = random.choice(["4k", "8k"])
    elif cat == "headphones":
        attrs["type"] = random.choice(["over-ear", "in-ear", "on-ear"])
        attrs["wireless"] = True
    elif cat == "phone":
        attrs["storage"] = random.choice([128, 256, 512, 1024])
        attrs["ram"] = random.choice([6, 8, 12, 16])
        attrs["screen"] = random.choice(["oled", "amoled", "lcd"])
    elif cat == "camera":
        attrs["sensor"] = random.choice(["aps-c", "full-frame", "micro-4-3"])
        attrs["video"] = random.choice(["4k", "6k", "8k"])

    return attrs


def generate_items(cfg: SimConfig) -> pd.DataFrame:
    _rng(cfg.seed)

    seller_ids = [f"S{str(i).zfill(5)}" for i in range(cfg.n_sellers)]
    cats = list(CATEGORIES.keys())

    rows = []
    for i in range(cfg.n_items):
        cat = random.choice(cats)
        meta = CATEGORIES[cat]
        brand = random.choice(meta["brands"])
        model = random.choice(meta["models"])

        price = _sample_price(*meta["price_range"])

        # stock y ventas con algo de sentido
        available = int(np.clip(np.random.poisson(lam=15), 0, 200))
        # vendidas correlacionan inversamente con precio (un poco) + ruido
        sold_base = max(0.0, (meta["price_range"][1] - price) / (meta["price_range"][1] - meta["price_range"][0] + 1e-9))
        sold = int(np.clip(np.random.poisson(lam=2 + 18 * sold_base), 0, 500))

        tags = random.sample(meta["tags"], k=min(len(meta["tags"]), random.choice([2, 3, 4])))
        # mete algunos tags "cruzados" para que embeddings tengan variedad
        if random.random() < 0.12:
            tags.append("oferta")
        if random.random() < 0.08:
            tags.append("premium")

        attrs = _make_attributes(cat, brand, model)

        title = f"{brand} {model} {cat}"
        if cat == "laptop":
            title = f"{brand} {model} Laptop {attrs['cpu']} {attrs['ram']}GB {attrs['storage']}GB"
        elif cat == "tv":
            title = f"{brand} {model} TV {attrs['screen_size']}\" {attrs['resolution']}"
        elif cat == "headphones":
            title = f"{brand} {model} Headphones Bluetooth"
        elif cat == "phone":
            title = f"{brand} {model} Phone {attrs['storage']}GB {attrs['ram']}GB"
        elif cat == "camera":
            title = f"{brand} {model} Camera {attrs['sensor']} {attrs['video']}"

        rows.append(
            {
                "item_id": f"I{str(i).zfill(7)}",
                "title": title,
                "seller_id": random.choice(seller_ids),
                "price": float(round(price, 2)),
                "available_quantity": available,
                "sold_quantity": sold,
                "tags": tags,
                "attributes": attrs,
            }
        )

    return pd.DataFrame(rows)


def generate_search_events(cfg: SimConfig, df_items: pd.DataFrame) -> pd.DataFrame:
    _rng(cfg.seed + 1)

    base_queries = [
        "laptop gamer i7 16gb hasta {budget}",
        "laptop trabajo ultrabook ssd hasta {budget}",
        "tv 4k 55 pulgadas barato hasta {budget}",
        "audifonos bluetooth cancelacion ruido hasta {budget}",
        "celular 5g buena camara hasta {budget}",
        "camara vlog 4k estabilizacion hasta {budget}",
    ]

    budgets = [300, 500, 800, 1200, 2000, 3000, 5000, 8000, 12000]
    start = datetime.utcnow() - timedelta(days=30)

    # para simular "click", tomamos un item razonable por categoría y precio
    df_items_small = df_items[["item_id", "price", "attributes"]].copy()

    def pick_clicked_item(query: str, budget: float) -> str | None:
        # categoría aproximada por keywords
        q = query.lower()
        if "tv" in q:
            cat = "tv"
        elif "audif" in q or "head" in q:
            cat = "headphones"
        elif "celular" in q or "phone" in q:
            cat = "phone"
        elif "camara" in q or "camera" in q:
            cat = "camera"
        else:
            cat = "laptop"

        # filtra por cat + budget
        mask_cat = df_items_small["attributes"].apply(lambda d: d.get("category") == cat)
        cand = df_items_small[mask_cat & (df_items_small["price"] <= budget)]
        if cand.empty:
            return None
        return str(cand.sample(1, random_state=int(budget)).iloc[0]["item_id"])

    rows = []
    for i in range(cfg.n_events):
        budget = float(random.choice(budgets))
        q_template = random.choice(base_queries)
        query = q_template.format(budget=int(budget))

        ts = start + timedelta(minutes=random.randint(0, 30 * 24 * 60))
        clicked = pick_clicked_item(query, budget)

        rows.append(
            {
                "event_id": f"E{str(i).zfill(7)}",
                "timestamp": ts.isoformat(),
                "user_id": f"U{str(random.randint(0, 9999)).zfill(4)}",
                "query": query,
                "budget": budget,
                "clicked_item_id": clicked,
            }
        )

    return pd.DataFrame(rows)


def save_artifacts(df_items: pd.DataFrame, df_events: pd.DataFrame, out_dir: str = "artifacts") -> None:
    df_items.to_parquet(f"{out_dir}/items.parquet", index=False)
    df_events.to_parquet(f"{out_dir}/events.parquet", index=False)


def main():
    cfg = SimConfig()
    items = generate_items(cfg)
    events = generate_search_events(cfg, items)
    save_artifacts(items, events)
    print("Saved:")
    print(" - artifacts/items.parquet")
    print(" - artifacts/events.parquet")
    print(items.head(3).to_string(index=False))
    print(events.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
