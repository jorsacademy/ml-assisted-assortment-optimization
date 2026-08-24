import numpy as np

from ml_assortment.data import generate_synthetic_retail_data
from ml_assortment.model import train_demand_model, predict_next_month_units
from ml_assortment.optimization import optimize_store_assortment


def test_optimizer_respects_constraints_and_does_not_hurt_baseline():
    data = generate_synthetic_retail_data(n_stores=10, n_products=32, n_months=16, seed=7)
    model_result = train_demand_model(
        data.stores,
        data.products,
        data.history,
        data.current_assortment,
    )
    scores = predict_next_month_units(model_result, data.stores, data.products, next_month=16)
    store = data.stores.iloc[0]
    result = optimize_store_assortment(
        store_id=store["store_id"],
        store_row=store,
        product_scores=scores,
        current_assortment=data.current_assortment,
        max_changes=6,
    )
    assert result.selected_product_ids
    assert result.changes <= 6
    assert result.objective_value + 1e-8 >= result.current_objective_value

    selected = data.products[data.products["product_id"].isin(result.selected_product_ids)]
    assert not selected["delisted"].any()
    assert selected["space_units"].sum() <= store["space_capacity"]

    current_count = data.current_assortment.query("store_id == @store.store_id")["in_assortment"].sum()
    assert len(result.selected_product_ids) == current_count


def test_pipeline_is_deterministic_for_fixed_seed():
    a = generate_synthetic_retail_data(n_stores=4, n_products=20, n_months=12, seed=123)
    b = generate_synthetic_retail_data(n_stores=4, n_products=20, n_months=12, seed=123)
    assert np.allclose(a.stores["footfall_index"], b.stores["footfall_index"])
    assert a.products.equals(b.products)
    assert a.current_assortment.equals(b.current_assortment)
    assert a.history.equals(b.history)
