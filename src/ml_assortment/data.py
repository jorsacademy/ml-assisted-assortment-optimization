from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticRetailData:
    stores: pd.DataFrame
    products: pd.DataFrame
    history: pd.DataFrame
    current_assortment: pd.DataFrame


def generate_synthetic_retail_data(
    n_stores: int = 80,
    n_products: int = 60,
    n_months: int = 24,
    seed: int = 42,
) -> SyntheticRetailData:
    """Generate internally consistent synthetic retail data.

    Revenue is produced from store demand, product attractiveness, seasonality,
    assortment availability, price sensitivity, and random shocks. No external
    company data is used.
    """
    rng = np.random.default_rng(seed)

    store_ids = [f"S{i:03d}" for i in range(n_stores)]
    product_ids = [f"P{i:03d}" for i in range(n_products)]
    categories = np.array(["Core", "Impulse", "Premium", "Value"])

    stores = pd.DataFrame({
        "store_id": store_ids,
        "footfall_index": rng.lognormal(mean=0.0, sigma=0.35, size=n_stores),
        "income_index": rng.normal(1.0, 0.18, size=n_stores).clip(0.55, 1.55),
        "space_capacity": rng.integers(22, 36, size=n_stores),
        "segment": rng.choice(["Urban", "Neighborhood", "Transit"], size=n_stores, p=[0.45, 0.4, 0.15]),
    })

    products = pd.DataFrame({
        "product_id": product_ids,
        "category": rng.choice(categories, size=n_products, p=[0.35, 0.25, 0.2, 0.2]),
        "price": rng.uniform(2.0, 14.0, size=n_products).round(2),
        "unit_margin": rng.uniform(0.7, 5.0, size=n_products).round(2),
        "space_units": rng.integers(1, 4, size=n_products),
        "base_attractiveness": rng.lognormal(mean=0.0, sigma=0.45, size=n_products),
        "delisted": False,
    })
    if n_products >= 10:
        products.loc[rng.choice(n_products, size=max(2, n_products // 20), replace=False), "delisted"] = True

    current_rows = []
    for _, store in stores.iterrows():
        allowed = products.loc[~products["delisted"]].copy()
        # The baseline is intentionally plausible but imperfect: local teams
        # favor attractive products while retaining substantial judgment noise.
        score = (
            0.55 * np.log1p(allowed["base_attractiveness"].to_numpy())
            + 0.20 * (allowed["unit_margin"].to_numpy() / allowed["unit_margin"].max())
            + rng.normal(0.0, 0.45, len(allowed))
        )
        order = np.argsort(score)[::-1]
        used = 0
        chosen: list[str] = []
        for idx in order:
            p = allowed.iloc[idx]
            if used + int(p["space_units"]) <= int(store["space_capacity"]):
                chosen.append(str(p["product_id"]))
                used += int(p["space_units"])
            if len(chosen) >= 18:
                break
        chosen_set = set(chosen)
        current_rows.extend(
            {"store_id": store["store_id"], "product_id": pid, "in_assortment": int(pid in chosen_set)}
            for pid in product_ids
        )
    current = pd.DataFrame(current_rows)

    assortment_lookup = current.set_index(["store_id", "product_id"])["in_assortment"].to_dict()
    history_rows = []
    for month in range(n_months):
        seasonal = 1.0 + 0.16 * np.sin(2 * np.pi * month / 12.0)
        market_shock = rng.normal(1.0, 0.04)
        for _, s in stores.iterrows():
            store_factor = float(s["footfall_index"]) * (0.75 + 0.25 * float(s["income_index"]))
            for _, p in products.iterrows():
                in_assortment = assortment_lookup[(s["store_id"], p["product_id"])]
                if not in_assortment or bool(p["delisted"]):
                    units = 0.0
                else:
                    affordability = np.exp(-0.035 * float(p["price"]) / float(s["income_index"]))
                    category_boost = {
                        "Urban": {"Premium": 1.20, "Impulse": 1.10},
                        "Neighborhood": {"Core": 1.15, "Value": 1.10},
                        "Transit": {"Impulse": 1.30, "Core": 1.05},
                    }.get(str(s["segment"]), {}).get(str(p["category"]), 1.0)
                    lam = 5.0 * store_factor * float(p["base_attractiveness"]) * affordability * category_boost * seasonal * market_shock
                    units = float(rng.poisson(max(lam, 0.05)))
                history_rows.append({
                    "month": month,
                    "store_id": s["store_id"],
                    "product_id": p["product_id"],
                    "units": units,
                    "revenue": units * float(p["price"]),
                })
    history = pd.DataFrame(history_rows)
    return SyntheticRetailData(stores=stores, products=products, history=history, current_assortment=current)
