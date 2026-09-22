from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProductSpec:
    name: str
    amplitude: float
    half_saturation: float
    hill_power: float = 1.25


def hill_response(
    treatment: np.ndarray | float,
    amplitude: float,
    half_saturation: float,
    hill_power: float,
) -> np.ndarray:
    """Saturating nonlinear treatment response."""
    t = np.asarray(treatment, dtype=float)
    numerator = np.power(np.clip(t, 0.0, None), hill_power)
    denominator = np.power(half_saturation, hill_power) + numerator
    return amplitude * numerator / denominator


def simulate_product(
    spec: ProductSpec,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate observational data with confounding and nonlinear treatment effects."""
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.uniform(-1.0, 1.0, n)
    x3 = rng.beta(2.0, 5.0, n)
    x4 = rng.normal(0.0, 0.8, n)

    latent_discount = (
        4_000
        + 850 * x1
        - 1_100 * x2
        + 2_200 * x3
        + 500 * x4
        + rng.normal(0.0, 900.0, n)
    )
    treatment = np.clip(latent_discount, 250.0, 10_000.0)

    baseline = (
        10.0
        + 2.5 * np.sin(x1)
        + 1.4 * np.square(x2)
        + 2.2 * x3
        + 0.8 * x1 * x4
    )

    treatment_effect = hill_response(
        treatment,
        amplitude=spec.amplitude,
        half_saturation=spec.half_saturation,
        hill_power=spec.hill_power,
    )

    y = baseline + treatment_effect + rng.normal(0.0, 0.45, n)

    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "treatment": treatment,
            "outcome": y,
            "true_effect": treatment_effect,
            "product": spec.name,
        }
    )
