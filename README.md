# ML-Assisted Assortment Optimization

A reproducible research project demonstrating how machine-learning demand estimates can feed a mixed-integer assortment optimization model.

This repository uses **fully synthetic retail data**. It is not based on any named retailer, brand, confidential dataset, or proprietary business rule.

## What the project does

1. Generates synthetic store, product, assortment, and monthly sales data.
2. Trains a nonlinear demand model using a time-based holdout.
3. Produces next-month store-product demand, revenue, and margin estimates.
4. Solves a binary mixed-integer optimization problem for every store.
5. Enforces shelf-space capacity, assortment-size stability, category guardrails, delisted-item exclusion, and a maximum number of assortment changes.
6. Compares the optimized assortment with the current assortment using predicted gross margin.

## Why this formulation is realistic

The optimization objective uses **product-level predicted economics**, not a store-level prediction duplicated across every SKU. The ML layer therefore produces coefficients that are meaningful for assortment decisions. The optimization layer then selects binary SKU decisions subject to operational constraints.

The demand model is estimated only from observations where a product was actually offered. Structural zeros created by unavailable products are excluded from demand-model training. This matters because an unavailable item is not evidence of zero underlying demand.

This is intentionally simpler than embedding a neural network directly inside a MILP. That formulation is possible, but it is substantially harder to validate and is unnecessary for demonstrating the core decision-support architecture.

## Mathematical model

For store `s` and product `p`, let `x_sp` be 1 if the product is selected and 0 otherwise. Let `m_sp` be predicted contribution margin and `a_sp` be the current assortment indicator.

Objective:

```text
maximize  sum_p m_sp * x_sp
```

Subject to:

```text
sum_p space_p * x_sp <= store_capacity_s
sum_p x_sp = sum_p a_sp
sum_p |x_sp - a_sp| <= max_changes
x_sp = 0                         for delisted products
category_min_c <= sum_{p in c} x_sp <= category_max_c
x_sp in {0, 1}
```

Absolute deviations are represented with auxiliary binary variables and linear equality constraints.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

## Run

```bash
PYTHONPATH=src python run_pipeline.py
```

Example with a smaller dataset:

```bash
PYTHONPATH=src python run_pipeline.py --stores 30 --products 40 --months 18 --max-changes 6
```

Outputs are written to `outputs/`:

- `model_metrics.json`
- `predicted_product_scores.csv`
- `optimization_results.csv`

## Tests

```bash
pip install -e '.[dev]'
pytest
```

## Validation status

The repository code is exercised with deterministic unit tests that check assortment cardinality, delisted-product exclusion, shelf-space capacity, change limits, baseline non-degradation under the optimization objective, and reproducible synthetic-data generation.

A local end-to-end run with the default configuration completed successfully before publication. Because the data is synthetic, numerical performance and predicted uplift are examples of simulated behavior rather than empirical business results.

## Important limitations

- Data is synthetic, so reported uplift is a simulation result, not a business claim.
- Predicted uplift is not causal uplift. A real deployment requires an experiment or credible causal design.
- The model does not account for substitution, cannibalization, stockouts, supplier constraints, or multi-period inventory decisions.
- Commercial deployment would require calibration with real transactional data and additional validation.

## License

This project is released under a custom **Research and Educational Use Only License**. Commercial use is prohibited. See `LICENSE`.
