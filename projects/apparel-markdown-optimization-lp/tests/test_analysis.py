import numpy as np
import pytest

from markdown_optimizer import apply_discounts, category_summary, generate_sample_data


def test_apply_discounts_calculates_expected_columns() -> None:
    df = generate_sample_data(12, seed=5)
    discounts = np.zeros(len(df))
    result = apply_discounts(df, discounts)

    assert {"discount", "discounted_price", "new_margin", "margin_impact", "new_inventory_value"}.issubset(result.columns)
    assert np.allclose(result["discounted_price"], df["current_price"])
    assert np.allclose(result["margin_impact"], 0.0)


def test_apply_discounts_rejects_wrong_length() -> None:
    df = generate_sample_data(10)
    with pytest.raises(ValueError, match="match df length"):
        apply_discounts(df, np.zeros(9))


def test_apply_discounts_rejects_below_cost_price() -> None:
    df = generate_sample_data(5, seed=9)
    with pytest.raises(ValueError, match="below cost"):
        apply_discounts(df, np.full(len(df), 0.90))


def test_category_summary_returns_one_row_per_category_present() -> None:
    df = generate_sample_data(30, seed=2)
    result = apply_discounts(df, np.zeros(len(df)))
    summary = category_summary(result)
    assert set(summary.index) == set(df["category"])
