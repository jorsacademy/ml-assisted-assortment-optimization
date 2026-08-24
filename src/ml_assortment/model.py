from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class DemandModelResult:
    model: Pipeline
    metrics: dict[str, float]
    feature_frame: pd.DataFrame


def build_training_frame(
    stores: pd.DataFrame,
    products: pd.DataFrame,
    history: pd.DataFrame,
    current_assortment: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = history.merge(stores, on="store_id", how="left").merge(products, on="product_id", how="left")
    if current_assortment is not None:
        df = df.merge(current_assortment, on=["store_id", "product_id"], how="left")
        # Estimate demand conditional on the product being offered. Structural
        # zeros from unavailable products are not valid demand observations.
        df = df[df["in_assortment"] == 1].copy()
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["target_log_units"] = np.log1p(df["units"])
    return df


def train_demand_model(
    stores: pd.DataFrame,
    products: pd.DataFrame,
    history: pd.DataFrame,
    current_assortment: pd.DataFrame | None = None,
) -> DemandModelResult:
    df = build_training_frame(stores, products, history, current_assortment=current_assortment)
    cutoff = int(df["month"].max()) - 3
    train = df[df["month"] <= cutoff].copy()
    test = df[df["month"] > cutoff].copy()

    numeric = [
        "footfall_index",
        "income_index",
        "price",
        "unit_margin",
        "space_units",
        "base_attractiveness",
        "month_sin",
        "month_cos",
    ]
    categorical = ["segment", "category"]
    features = numeric + categorical

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("num", "passthrough", numeric),
    ])
    reg = HistGradientBoostingRegressor(max_depth=8, learning_rate=0.08, max_iter=220, random_state=42)
    pipe = Pipeline([("prep", preprocess), ("reg", reg)])
    pipe.fit(train[features], train["target_log_units"])

    pred_units = np.expm1(pipe.predict(test[features])).clip(0.0)
    y_true = test["units"].to_numpy()
    metrics = {
        "mae_units": float(mean_absolute_error(y_true, pred_units)),
        "r2_units": float(r2_score(y_true, pred_units)),
    }
    return DemandModelResult(model=pipe, metrics=metrics, feature_frame=df)


def predict_next_month_units(
    result: DemandModelResult,
    stores: pd.DataFrame,
    products: pd.DataFrame,
    next_month: int,
) -> pd.DataFrame:
    cross = stores.assign(_key=1).merge(products.assign(_key=1), on="_key").drop(columns="_key")
    cross["month"] = next_month
    cross["month_sin"] = np.sin(2 * np.pi * next_month / 12.0)
    cross["month_cos"] = np.cos(2 * np.pi * next_month / 12.0)
    numeric = [
        "footfall_index",
        "income_index",
        "price",
        "unit_margin",
        "space_units",
        "base_attractiveness",
        "month_sin",
        "month_cos",
    ]
    categorical = ["segment", "category"]
    cross["predicted_units"] = np.expm1(result.model.predict(cross[numeric + categorical])).clip(0.0)
    cross["predicted_revenue"] = cross["predicted_units"] * cross["price"]
    cross["predicted_margin"] = cross["predicted_units"] * cross["unit_margin"]
    return cross
