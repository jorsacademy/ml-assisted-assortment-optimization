from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix


@dataclass
class OptimizationResult:
    store_id: str
    selected_product_ids: list[str]
    objective_value: float
    current_objective_value: float
    predicted_lift_pct: float
    changes: int
    status: str


def optimize_store_assortment(
    store_id: str,
    store_row: pd.Series,
    product_scores: pd.DataFrame,
    current_assortment: pd.DataFrame,
    max_changes: int = 6,
    objective: str = "predicted_margin",
) -> OptimizationResult:
    df = product_scores[product_scores["store_id"] == store_id].reset_index(drop=True).copy()
    n = len(df)
    if n == 0:
        raise ValueError(f"No product scores available for store {store_id}")

    current_map = current_assortment[current_assortment["store_id"] == store_id].set_index("product_id")["in_assortment"].to_dict()
    current = np.array([int(current_map.get(pid, 0)) for pid in df["product_id"]], dtype=float)
    scores = df[objective].to_numpy(dtype=float)
    space = df["space_units"].to_numpy(dtype=float)
    delisted = df["delisted"].astype(int).to_numpy()
    capacity = float(store_row["space_capacity"])

    # Variables are x_j (selection) and d_j (absolute change indicator).
    c = np.concatenate([-scores, np.zeros(n)])
    integrality = np.ones(2 * n)
    lb = np.zeros(2 * n)
    ub = np.ones(2 * n)
    ub[:n][delisted == 1] = 0.0

    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    # Shelf-space capacity.
    r = np.zeros(2 * n)
    r[:n] = space
    rows.append(r)
    lower.append(-np.inf)
    upper.append(capacity)

    # Keep assortment cardinality unchanged.
    r = np.zeros(2 * n)
    r[:n] = 1.0
    rows.append(r)
    lower.append(float(current.sum()))
    upper.append(float(current.sum()))

    # Exact absolute-deviation linearization for binary x and fixed binary current:
    # d_j = |x_j - current_j|.
    for j in range(n):
        r = np.zeros(2 * n)
        r[j] = -1.0 if current[j] == 0 else 1.0
        r[n + j] = 1.0
        rhs = 0.0 if current[j] == 0 else 1.0
        rows.append(r)
        lower.append(rhs)
        upper.append(rhs)

    r = np.zeros(2 * n)
    r[n:] = 1.0
    rows.append(r)
    lower.append(-np.inf)
    upper.append(float(max_changes))

    # Segment-aware category guardrails. Requirements are capped by the current
    # feasible assortment so the model does not become infeasible solely because
    # the baseline assortment lacks a category.
    segment_rules = {
        "Urban": {"Core": (2, None), "Impulse": (2, None), "Premium": (1, 7), "Value": (1, None)},
        "Neighborhood": {"Core": (3, None), "Impulse": (1, None), "Premium": (0, 5), "Value": (2, None)},
        "Transit": {"Core": (2, None), "Impulse": (3, None), "Premium": (0, 5), "Value": (1, None)},
    }
    requested_rules = segment_rules.get(str(store_row["segment"]), segment_rules["Urban"])
    categories = df["category"].to_numpy()
    for cat, (requested_min, requested_max) in requested_rules.items():
        idx = np.where(categories == cat)[0]
        if len(idx) == 0:
            continue
        available = int(np.sum(delisted[idx] == 0))
        current_count = int(current[idx].sum())
        effective_min = min(int(requested_min), current_count, available)
        effective_max = np.inf if requested_max is None else float(min(int(requested_max), available))
        if effective_max != np.inf and current_count > effective_max:
            effective_max = float(current_count)
        r = np.zeros(2 * n)
        r[idx] = 1.0
        rows.append(r)
        lower.append(float(effective_min))
        upper.append(effective_max)

    constraint = LinearConstraint(csr_matrix(np.vstack(rows)), np.array(lower), np.array(upper))
    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=constraint,
        options={"time_limit": 30.0},
    )
    if not result.success or result.x is None:
        return OptimizationResult(
            store_id=store_id,
            selected_product_ids=[],
            objective_value=float("nan"),
            current_objective_value=float(np.dot(scores, current)),
            predicted_lift_pct=float("nan"),
            changes=0,
            status=result.message,
        )

    x = np.rint(result.x[:n]).astype(int)
    selected = df.loc[x == 1, "product_id"].tolist()
    new_obj = float(np.dot(scores, x))
    current_obj = float(np.dot(scores, current))
    lift = 100.0 * (new_obj - current_obj) / current_obj if current_obj > 0 else float("nan")
    changes = int(np.abs(x - current).sum())
    return OptimizationResult(
        store_id=store_id,
        selected_product_ids=selected,
        objective_value=new_obj,
        current_objective_value=current_obj,
        predicted_lift_pct=lift,
        changes=changes,
        status=result.message,
    )
