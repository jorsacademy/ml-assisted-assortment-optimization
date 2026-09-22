from __future__ import annotations

import numpy as np

from causal_pricing.curves import fit_hill_curve
from causal_pricing.optimizer import optimize_budget
from causal_pricing.simulation import ProductSpec, simulate_product
from causal_pricing.slearner import average_counterfactual_response, fit_slearner


def main() -> None:
    rng = np.random.default_rng(20260828)

    product_specs = [
        ProductSpec("product_a", amplitude=3.2, half_saturation=3_600, hill_power=1.20),
        ProductSpec("product_b", amplitude=1.4, half_saturation=5_400, hill_power=1.45),
        ProductSpec("product_c", amplitude=5.1, half_saturation=3_000, hill_power=1.10),
    ]

    learned_curves = {}
    observed_means = {}

    for spec in product_specs:
        data = simulate_product(spec, n=35_000, rng=rng)
        observed_means[spec.name] = float(data["treatment"].mean())

        learner = fit_slearner(data)

        q_low, q_high = data["treatment"].quantile([0.02, 0.98])
        treatment_grid = np.linspace(float(q_low), float(q_high), 80)

        response = average_counterfactual_response(
            learner,
            treatment_grid=treatment_grid,
            max_profiles=3_000,
        )
        params = fit_hill_curve(response)
        learned_curves[spec.name] = params

        print(
            f"{spec.name}: "
            f"test MSE={learner.test_mse:.4f}, "
            f"test R2={learner.test_r2:.4f}, "
            f"curve=(A={params.amplitude:.3f}, "
            f"K={params.half_saturation:.1f}, "
            f"h={params.hill_power:.3f})"
        )

    historical_budget = sum(observed_means.values())
    total_budget = 0.80 * historical_budget

    bounds = {
        name: (0.40 * mean_budget, 1.35 * mean_budget)
        for name, mean_budget in observed_means.items()
    }

    result = optimize_budget(
        parameters=learned_curves,
        total_budget=total_budget,
        bounds=bounds,
    )

    print("\nHistorical total budget:", round(historical_budget, 2))
    print("New total budget:", round(total_budget, 2))
    print("Optimal allocation:")

    for product, value in result.allocation.items():
        print(f"  {product}: {value:,.2f}")

    print(
        "Expected incremental orders:",
        round(result.expected_incremental_orders, 4),
    )
    print("Budget residual:", f"{result.budget_residual:.3e}")


if __name__ == "__main__":
    main()
