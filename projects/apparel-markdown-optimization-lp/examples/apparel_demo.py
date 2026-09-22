from markdown_optimizer import (
    apply_discounts,
    category_summary,
    generate_sample_data,
    optimize_discounts,
)


def main() -> None:
    inventory = generate_sample_data(n_products=100, seed=42)
    initial_value = float(inventory["inventory_value"].sum())
    initial_margin = float(
        (inventory["margin"] * inventory["inventory_value"]).sum() / initial_value
    )

    target_inventory_value = initial_value * 0.90
    discounts = optimize_discounts(
        inventory,
        target_inventory_value=target_inventory_value,
        max_discount=0.50,
        min_discount=0.0,
        category_average_cap=0.30,
    )
    result = apply_discounts(inventory, discounts)

    new_value = float(result["new_inventory_value"].sum())
    new_margin = float(
        (result["new_margin"] * result["new_inventory_value"]).sum() / new_value
    )

    print(f"Initial inventory value: ${initial_value:,.2f}")
    print(f"Target post-markdown value: ${target_inventory_value:,.2f}")
    print(f"Actual post-markdown value: ${new_value:,.2f}")
    print(f"Initial weighted margin: {initial_margin:.2%}")
    print(f"Post-markdown weighted margin: {new_margin:.2%}")
    print(f"Margin impact: {new_margin - initial_margin:.2%}")
    print("\nCategory summary:")
    print(category_summary(result))


if __name__ == "__main__":
    main()
