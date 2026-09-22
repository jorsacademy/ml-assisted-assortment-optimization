from __future__ import annotations

import numpy as np
import pandas as pd

from .optimizer import validate_inventory_data


def apply_discounts(df: pd.DataFrame, discount_array: np.ndarray) -> pd.DataFrame:
    """Apply optimized discounts and calculate post-markdown metrics."""
    validate_inventory_data(df)

    discounts = np.asarray(discount_array, dtype=float)
    if discounts.ndim != 1 or len(discounts) != len(df):
        raise ValueError("discount_array must be one-dimensional and match df length")
    if not np.isfinite(discounts).all():
        raise ValueError("discount_array must contain only finite values")
    if ((discounts < 0) | (discounts >= 1)).any():
        raise ValueError("discounts must satisfy 0 <= discount < 1")

    result = df.copy()
    result["discount"] = discounts
    result["discounted_price"] = result["current_price"] * (1.0 - result["discount"])

    if (result["discounted_price"] + 1e-9 < result["cost"]).any():
        raise ValueError("discount_array would price at least one SKU below cost")

    result["new_margin"] = (
        result["discounted_price"] - result["cost"]
    ) / result["discounted_price"]

    if "margin" not in result.columns:
        result["margin"] = (
            result["current_price"] - result["cost"]
        ) / result["current_price"]

    result["margin_impact"] = result["new_margin"] - result["margin"]
    result["new_inventory_value"] = result["discounted_price"] * result["inventory"]
    return result


def category_summary(result: pd.DataFrame) -> pd.DataFrame:
    """Summarize markdown outcomes by apparel category."""
    required = {"category", "discount", "margin_impact", "inventory", "new_inventory_value"}
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f"missing result columns: {sorted(missing)}")

    return result.groupby("category", sort=True).agg(
        average_discount=("discount", "mean"),
        average_margin_impact=("margin_impact", "mean"),
        inventory_units=("inventory", "sum"),
        new_inventory_value=("new_inventory_value", "sum"),
    )
