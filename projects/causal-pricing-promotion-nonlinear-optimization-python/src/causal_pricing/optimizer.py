from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .curves import HillParameters, hill_curve


@dataclass(frozen=True)
class AllocationResult:
    allocation: dict[str, float]
    expected_incremental_orders: float
    budget_residual: float
    success: bool
    message: str


def optimize_budget(
    parameters: dict[str, HillParameters],
    total_budget: float,
    bounds: dict[str, tuple[float, float]],
) -> AllocationResult:
    """Maximize total incremental response under a shared budget."""
    products = list(parameters)

    if set(products) != set(bounds):
        raise ValueError("parameters and bounds must contain the same product names")

    lower_sum = sum(bounds[p][0] for p in products)
    upper_sum = sum(bounds[p][1] for p in products)

    if not lower_sum <= total_budget <= upper_sum:
        raise ValueError(
            f"Infeasible budget: {total_budget:.2f}; "
            f"feasible interval is [{lower_sum:.2f}, {upper_sum:.2f}]"
        )

    def total_response(x: np.ndarray) -> float:
        total = 0.0
        for product, budget in zip(products, x, strict=True):
            p = parameters[product]
            total += float(
                hill_curve(
                    budget,
                    p.amplitude,
                    p.half_saturation,
                    p.hill_power,
                )
            )
        return total

    def objective(x: np.ndarray) -> float:
        return -total_response(x)

    scipy_bounds = [bounds[p] for p in products]

    x0 = np.array([total_budget / len(products)] * len(products), dtype=float)
    x0 = np.array(
        [np.clip(value, low, high) for value, (low, high) in zip(x0, scipy_bounds)],
        dtype=float,
    )

    residual = total_budget - float(x0.sum())
    if abs(residual) > 1e-10:
        for i, (low, high) in enumerate(scipy_bounds):
            room = high - x0[i] if residual > 0 else x0[i] - low
            move = np.sign(residual) * min(abs(residual), room)
            x0[i] += move
            residual -= move
            if abs(residual) < 1e-10:
                break

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=scipy_bounds,
        constraints={
            "type": "eq",
            "fun": lambda x: float(np.sum(x) - total_budget),
        },
        options={
            "maxiter": 2_000,
            "ftol": 1e-10,
            "disp": False,
        },
    )

    budget_residual = float(np.sum(result.x) - total_budget)

    if not result.success:
        raise RuntimeError(f"SLSQP failed: {result.message}")

    if abs(budget_residual) > 1e-5:
        raise RuntimeError(f"Budget constraint residual too large: {budget_residual}")

    return AllocationResult(
        allocation={
            product: float(value)
            for product, value in zip(products, result.x, strict=True)
        },
        expected_incremental_orders=total_response(result.x),
        budget_residual=budget_residual,
        success=bool(result.success),
        message=str(result.message),
    )
