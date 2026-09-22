# Apparel Markdown Optimization with Linear Programming

A small, reproducible Python project for allocating markdown discounts across apparel SKUs with linear programming.

> **NON-COMMERCIAL USE ONLY.** Commercial use, paid consulting use, SaaS/API integration, internal for-profit operational use, resale, sublicensing, and incorporation into commercial products are prohibited unless separately licensed in writing by the copyright holder. See `LICENSE.md`.

## What this model does

The optimizer allocates a required total markdown value across products while respecting SKU-level and category-level guardrails. It prioritizes products that are older, carry more inventory, and sell more slowly.

The linear program uses discount fraction `d_i` as the decision variable for each SKU.

The markdown-value target is enforced as:

```text
sum(current_price_i * inventory_i * d_i)
    = initial_inventory_value - target_inventory_value
```

The objective maximizes markdown dollars assigned to higher-priority SKUs, where priority is a normalized weighted score based on days in stock, inventory, and inverse sales velocity.

## Important modeling limitation

This project does **not** claim that a discount causes a particular increase in unit demand. No price-elasticity or demand-response curve is present in the supplied data. Therefore this is a **markdown allocation model**, not a demand-elasticity or inventory-clearance forecasting model.

A production pricing system should add estimated price elasticity, demand forecasts, stockout effects, competitor pricing, seasonality, customer segmentation, and uncertainty before making operational pricing decisions.

## Guardrails

- Validates the required input schema and numeric domains.
- Rejects non-positive prices and sales velocity.
- Rejects cost above current price in the baseline dataset.
- Prevents the optimized discounted price from falling below cost.
- Supports minimum and maximum SKU discounts.
- Caps average discount by category.
- Checks solver success and raises a descriptive error for infeasible models.
- Generates deterministic synthetic apparel data for examples and tests.

## Project structure

```text
apparel-markdown-optimization-lp/
├── .github/workflows/ci.yml
├── examples/apparel_demo.py
├── src/markdown_optimizer/
│   ├── __init__.py
│   ├── analysis.py
│   ├── data.py
│   └── optimizer.py
├── tests/
│   ├── test_analysis.py
│   ├── test_data.py
│   └── test_optimizer.py
├── LICENSE.md
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
pytest
```

## Example

```python
from markdown_optimizer import (
    apply_discounts,
    generate_sample_data,
    optimize_discounts,
)

inventory = generate_sample_data(n_products=100, seed=42)
initial_value = inventory["inventory_value"].sum()

discounts = optimize_discounts(
    inventory,
    target_inventory_value=initial_value * 0.90,
    max_discount=0.50,
    category_average_cap=0.30,
)

result = apply_discounts(inventory, discounts)
print(result[["product_id", "category", "discount", "discounted_price"]].head())
```

Run the complete demonstration:

```bash
python examples/apparel_demo.py
```

## Input columns

The optimizer expects:

- `product_id`
- `category`
- `current_price`
- `cost`
- `inventory`
- `days_in_stock`
- `sales_velocity`

`generate_sample_data()` also adds baseline `margin` and `inventory_value` columns.

## Optimization formulation

For each product `i`, let `d_i` be its discount fraction and `v_i = current_price_i * inventory_i` its inventory exposure.

The model minimizes the negative priority-weighted markdown value:

```text
minimize -sum(priority_i * v_i * d_i)
```

subject to:

```text
sum(v_i * d_i) = required_markdown_value

average discount within each category <= category_average_cap

min_discount <= d_i <= min(max_discount, 1 - cost_i/current_price_i)
```

The final bound is the gross-margin floor: the discounted price cannot be lower than unit cost.

## CI

GitHub Actions runs tests on Python 3.10, 3.11, 3.12, and 3.13, plus Ruff linting.

## License

Copyright (c) 2026 JORS Academy.

This repository is source-available for non-commercial educational and research use only. See `LICENSE.md` for the complete terms. Commercial licensing requires separate written permission.
