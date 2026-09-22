import pandas as pd
import pytest

from markdown_optimizer import generate_sample_data


def test_sample_data_is_deterministic() -> None:
    left = generate_sample_data(25, seed=7)
    right = generate_sample_data(25, seed=7)
    pd.testing.assert_frame_equal(left, right)


def test_sample_data_has_positive_baseline_margin() -> None:
    df = generate_sample_data(100, seed=42)
    assert (df["current_price"] > df["cost"]).all()
    assert (df["margin"] > 0).all()
    assert (df["inventory_value"] > 0).all()


def test_sample_data_rejects_non_positive_size() -> None:
    with pytest.raises(ValueError):
        generate_sample_data(0)
