from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = ["x1", "x2", "x3", "x4", "treatment"]
COVARIATES = ["x1", "x2", "x3", "x4"]


@dataclass
class FittedSLearner:
    model: LGBMRegressor
    train: pd.DataFrame
    test: pd.DataFrame
    test_mse: float
    test_r2: float


def fit_slearner(data: pd.DataFrame, random_state: int = 42) -> FittedSLearner:
    """Fit a flexible single-model learner with treatment included as a feature."""
    train, test = train_test_split(
        data,
        test_size=0.25,
        random_state=random_state,
    )

    model = LGBMRegressor(
        objective="regression",
        n_estimators=700,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.9,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.15,
        random_state=random_state,
        verbosity=-1,
    )
    model.fit(train[FEATURES], train["outcome"])

    pred = model.predict(test[FEATURES])

    return FittedSLearner(
        model=model,
        train=train,
        test=test,
        test_mse=float(mean_squared_error(test["outcome"], pred)),
        test_r2=float(r2_score(test["outcome"], pred)),
    )


def average_counterfactual_response(
    fitted: FittedSLearner,
    treatment_grid: np.ndarray,
    max_profiles: int = 4_000,
    random_state: int = 123,
) -> pd.DataFrame:
    """Estimate an average incremental dose-response curve."""
    reference = fitted.test[COVARIATES].sample(
        n=min(max_profiles, len(fitted.test)),
        random_state=random_state,
        replace=False,
    )

    baseline_frame = reference.copy()
    baseline_frame["treatment"] = 0.0
    baseline_prediction = fitted.model.predict(baseline_frame[FEATURES])

    effects: list[float] = []

    for treatment in np.asarray(treatment_grid, dtype=float):
        scored = reference.copy()
        scored["treatment"] = treatment
        treated_prediction = fitted.model.predict(scored[FEATURES])
        effects.append(float(np.mean(treated_prediction - baseline_prediction)))

    return pd.DataFrame(
        {
            "treatment": np.asarray(treatment_grid, dtype=float),
            "estimated_effect": effects,
        }
    )
