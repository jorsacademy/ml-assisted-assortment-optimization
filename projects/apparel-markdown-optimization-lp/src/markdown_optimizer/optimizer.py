from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog


REQUIRED_COLUMNS = {
    "product_id",
    "category",
    "current_price",
    "cost",
    "inventory",
    "days_in_stock",
    "sales_velocity",
}


def validate_inventory_data(df: pd.DataFrame) -> None:
    """Validate the optimizer input schema and numerical domains."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if df.empty:
        raise ValueError("inventory data must not be empty")

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    numeric_columns = [
        "current_price",
        "cost",
        "inventory",
        "days_in_stock",
        "sales_velocity",
    ]
    if df[numeric_columns].isna().any().any():
        raise ValueError("numeric input columns must not contain missing values")

    if (df["current_price"] <= 0).any():
        raise ValueError("current_price must be positive")
    if (df["cost"] < 0).any():
        raise ValueError("cost must be non-negative")
    if (df["inventory"] < 0).any():
        raise ValueError("inventory must be non-negative")
    if (df["days_in_stock"] < 0).any():
        raise ValueError("days_in_stock must be non-negative")
    if (df["sales_velocity"] <= 0).any():
        raise ValueError("sales_velocity must be positive")
    if (df["cost"] > df["current_price"]).any():
        raise ValueError("cost must not exceed current_price")


def _minmax(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    span = values.max() - values.min()
    if span == 0:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    return (values - values.min()) / span


def clearance_priority(df: pd.DataFrame) -> pd.Series:
    """Return a normalized heuristic markdown-priority score in [0, 1]."""
    validate_inventory_data(df)

    age_score = _minmax(df["days_in_stock"])
    inventory_score = _minmax(df["inventory"])
    slow_mover_score = 1.0 - _minmax(df["sales_velocity"])

    return 0.45 * age_score + 0.30 * inventory_score + 0.25 * slow_mover_score


def optimize_discounts(
    df: pd.DataFrame,
    target_inventory_value: float,
    max_discount: float = 0.50,
    min_discount: float = 0.0,
    category_average_cap: float = 0.30,
) -> np.ndarray:
    """Allocate markdown discounts with a linear program.

    The target refers to the post-markdown inventory value under the simplifying
    assumption that inventory quantities themselves do not change inside this
    model. The LP allocates the exact required markdown value to higher-priority
    SKUs while observing discount and gross-margin guardrails.
    """
    validate_inventory_data(df)

    if not 0 <= min_discount <= max_discount < 1:
        raise ValueError("discount bounds must satisfy 0 <= min <= max < 1")
    if not 0 < category_average_cap < 1:
        raise ValueError("category_average_cap must be between 0 and 1")

    exposure = (df["current_price"] * df["inventory"]).to_numpy(dtype=float)
    initial_inventory_value = float(exposure.sum())

    if not 0 <= target_inventory_value <= initial_inventory_value:
        raise ValueError(
            "target_inventory_value must be between 0 and the initial inventory value"
        )

    required_markdown_value = initial_inventory_value - float(target_inventory_value)
    priority = clearance_priority(df).to_numpy(dtype=float)

    # scipy.optimize.linprog minimizes. Negative coefficients therefore maximize
    # priority-weighted markdown dollars.
    objective = -(priority * exposure)

    category_matrix = []
    category_limits = []
    categories = df["category"].to_numpy()
    for category in pd.unique(df["category"]):
        mask = (categories == category).astype(float)
        category_matrix.append(mask)
        category_limits.append(float(mask.sum()) * category_average_cap)

    bounds: list[tuple[float, float]] = []
    for price, cost in zip(df["current_price"], df["cost"], strict=True):
        margin_floor_cap = max(0.0, 1.0 - float(cost) / float(price))
        upper_bound = min(max_discount, margin_floor_cap)
        if upper_bound + 1e-12 < min_discount:
            raise ValueError(
                "min_discount conflicts with the gross-margin floor for at least one SKU"
            )
        bounds.append((min_discount, upper_bound))

    result = linprog(
        c=objective,
        A_ub=np.asarray(category_matrix, dtype=float),
        b_ub=np.asarray(category_limits, dtype=float),
        A_eq=np.asarray([exposure], dtype=float),
        b_eq=np.asarray([required_markdown_value], dtype=float),
        bounds=bounds,
        method="highs",
    )

    if not result.success or result.x is None:
        raise ValueError(f"discount optimization is infeasible: {result.message}")

    return np.asarray(result.x, dtype=float)
