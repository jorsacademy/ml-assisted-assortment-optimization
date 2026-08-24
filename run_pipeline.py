from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from ml_assortment.data import generate_synthetic_retail_data
from ml_assortment.model import train_demand_model, predict_next_month_units
from ml_assortment.optimization import optimize_store_assortment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stores", type=int, default=80)
    parser.add_argument("--products", type=int, default=60)
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-changes", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = generate_synthetic_retail_data(args.stores, args.products, args.months, args.seed)
    model_result = train_demand_model(
        data.stores,
        data.products,
        data.history,
        data.current_assortment,
    )
    next_scores = predict_next_month_units(
        model_result,
        data.stores,
        data.products,
        next_month=args.months,
    )

    results = []
    for _, store in data.stores.iterrows():
        res = optimize_store_assortment(
            store_id=str(store["store_id"]),
            store_row=store,
            product_scores=next_scores,
            current_assortment=data.current_assortment,
            max_changes=args.max_changes,
        )
        results.append(res.__dict__)

    pd.DataFrame(results).to_csv(args.output_dir / "optimization_results.csv", index=False)
    next_scores.to_csv(args.output_dir / "predicted_product_scores.csv", index=False)
    with open(args.output_dir / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(model_result.metrics, f, indent=2)

    print("Model metrics:", model_result.metrics)
    solved = pd.DataFrame(results)
    print("Mean predicted lift (%):", solved["predicted_lift_pct"].dropna().mean())
    print("Outputs written to", args.output_dir)


if __name__ == "__main__":
    main()
