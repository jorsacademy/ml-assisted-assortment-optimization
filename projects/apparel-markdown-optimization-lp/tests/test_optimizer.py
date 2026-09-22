import numpy as np
import pytest

from markdown_optimizer import generate_sample_data, optimize_discounts


def test_optimizer_hits_markdown_target_and_bounds() -> None:
    df = generate_sample_data(60, seed=42)
    exposure = (df["current_price"] * df["inventory"]).to_numpy(dtype=float)
    initial_value = float(exposure.sum())
    target = initial_value * 0.90

    discounts = optimize_discounts(
        df,
        target_inventory_value=target,
        max_discount=0.50,
        category_average_cap=0.30,
    )

    markdown_value = float(np.dot(exposure, discounts))
    assert markdown_value == pytest.approx(initial_value - target, rel=1e-8, abs=1e-6)
    assert (discounts >= -1e-10).all()
    assert (discounts <= 0.50 + 1e-10).all()


def test_optimizer_respects_category_average_cap() -> None:
    df = generate_sample_data(80, seed=11)
    initial_value = float(df["inventory_value"].sum())
    discounts = optimize_discounts(
        df,
        target_inventory_value=initial_value * 0.92,
        category_average_cap=0.20,
    )

    check = df[["category"]].copy()
    check["discount"] = discounts
    assert (check.groupby("category")["discount"].mean() <= 0.20 + 1e-9).all()


def test_optimizer_never_prices_below_cost() -> None:
    df = generate_sample_data(50, seed=4)
    initial_value = float(df["inventory_value"].sum())
    discounts = optimize_discounts(df, target_inventory_value=initial_value * 0.95)
    discounted_price = df["current_price"].to_numpy() * (1.0 - discounts)
    assert (discounted_price + 1e-9 >= df["cost"].to_numpy()).all()


def test_optimizer_rejects_invalid_sales_velocity() -> None:
    df = generate_sample_data(10)
    df.loc[0, "sales_velocity"] = 0.0
    with pytest.raises(ValueError, match="sales_velocity"):
        optimize_discounts(df, target_inventory_value=float(df["inventory_value"].sum()))


def test_optimizer_reports_infeasible_target() -> None:
    df = generate_sample_data(30, seed=3)
    with pytest.raises(ValueError, match="infeasible"):
        optimize_discounts(df, target_inventory_value=0.0, max_discount=0.10)
