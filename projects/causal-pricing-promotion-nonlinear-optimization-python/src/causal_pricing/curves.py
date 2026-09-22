from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class HillParameters:
    amplitude: float
    half_saturation: float
    hill_power: float


def hill_curve(
    treatment: np.ndarray | float,
    amplitude: float,
    half_saturation: float,
    hill_power: float,
) -> np.ndarray:
    t = np.asarray(treatment, dtype=float)
    numerator = np.power(np.clip(t, 0.0, None), hill_power)
    denominator = np.power(half_saturation, hill_power) + numerator
    return amplitude * numerator / denominator


def fit_hill_curve(response: pd.DataFrame) -> HillParameters:
    """Fit a monotone saturating response curve with positive parameters."""
    x = response["treatment"].to_numpy(dtype=float)
    y = response["estimated_effect"].to_numpy(dtype=float)

    amplitude0 = max(float(np.max(y)) * 1.15, 1e-3)
    half0 = max(float(np.median(x)), 1.0)
    power0 = 1.0

    params, _ = curve_fit(
        hill_curve,
        x,
        y,
        p0=(amplitude0, half0, power0),
        bounds=(
            (1e-8, 1e-8, 0.25),
            (np.inf, np.inf, 4.0),
        ),
        maxfev=50_000,
    )

    return HillParameters(
        amplitude=float(params[0]),
        half_saturation=float(params[1]),
        hill_power=float(params[2]),
    )
