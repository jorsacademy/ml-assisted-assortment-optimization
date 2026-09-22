from __future__ import annotations

import numpy as np
import pandas as pd


CATEGORIES = ("Shirts", "Pants", "Dresses", "Jackets", "Accessories")


def generate_sample_data(n_products: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate deterministic synthetic apparel inventory data.

    The generator intentionally keeps unit cost below current price so the
    baseline dataset starts with a positive gross margin for every SKU.
    """
    if n_products <= 0:
        raise ValueError("n_products must be a positive integer")

    rng = np.random.default_rng(seed)

    cost = rng.uniform(10.0, 100.0, n_products).round(2)
    markup_factor = rng.uniform(1.25, 2.50, n_products)
    current_price = np.maximum(cost * markup_factor, cost + 1.0).round(2)

    df = pd.DataFrame(
        {
            "product_id": np.arange(1, n_products + 1),
            "category": rng.choice(CATEGORIES, n_products),
            "current_price": current_price,
            "cost": cost,
            "inventory": rng.integers(5, 100, n_products),
            "days_in_stock": rng.integers(10, 120, n_products),
            "sales_velocity": rng.uniform(0.1, 5.0, n_products).round(2),
        }
    )

    df["margin"] = (df["current_price"] - df["cost"]) / df["current_price"]
    df["inventory_value"] = df["current_price"] * df["inventory"]
    return df
