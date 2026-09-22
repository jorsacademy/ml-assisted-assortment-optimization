from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class RevenueConfig:
    capacity: int = 80
    horizon: int = 60
    min_price: float = 80.0
    max_price: float = 320.0
    price_levels: int = 9
    reference_price: float = 180.0
    base_demand: float = 3.0
    price_sensitivity: float = 0.012
    time_growth: float = 0.9
    high_value_share: float = 0.25
    high_value_multiplier: float = 1.35
    spoilage_penalty: float = 18.0


class _RevenueCore(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: RevenueConfig | None = None):
        super().__init__()
        self.config = config or RevenueConfig()
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.remaining_capacity = self.config.capacity
        self.t = 0
        self.total_revenue = 0.0
        self.total_sales = 0
        self.last_price = self.config.reference_price

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.remaining_capacity = self.config.capacity
        self.t = 0
        self.total_revenue = 0.0
        self.total_sales = 0
        self.last_price = self.config.reference_price
        return self._obs(), self._info()

    def _obs(self) -> np.ndarray:
        time_remaining = (self.config.horizon - self.t) / self.config.horizon
        capacity_fraction = self.remaining_capacity / self.config.capacity
        progress = self.t / max(1, self.config.horizon - 1)
        last_price_scaled = (self.last_price - self.config.min_price) / (
            self.config.max_price - self.config.min_price
        )
        return np.array(
            [capacity_fraction, time_remaining, progress, np.clip(last_price_scaled, 0.0, 1.0)],
            dtype=np.float32,
        )

    def _expected_arrivals(self, price: float) -> float:
        progress = self.t / max(1, self.config.horizon - 1)
        time_factor = 1.0 + self.config.time_growth * progress
        elasticity = math.exp(-self.config.price_sensitivity * (price - self.config.reference_price))
        mix_boost = 1.0 + self.config.high_value_share * (
            self.config.high_value_multiplier - 1.0
        ) * progress
        return max(0.02, self.config.base_demand * time_factor * elasticity * mix_boost)

    def _step_price(self, price: float):
        price = float(np.clip(price, self.config.min_price, self.config.max_price))
        self.last_price = price
        expected = self._expected_arrivals(price)
        arrivals = int(self.np_random.poisson(expected))
        sales = min(arrivals, self.remaining_capacity)
        revenue = sales * price
        self.remaining_capacity -= sales
        self.total_sales += sales
        self.total_revenue += revenue
        self.t += 1

        terminated = self.remaining_capacity <= 0
        truncated = self.t >= self.config.horizon
        spoilage = 0.0
        if truncated and self.remaining_capacity > 0:
            spoilage = self.remaining_capacity * self.config.spoilage_penalty

        reward = revenue - spoilage
        info = self._info()
        info.update(
            {
                "price": price,
                "arrivals": arrivals,
                "sales": sales,
                "period_revenue": revenue,
                "expected_arrivals": expected,
                "spoilage_penalty": spoilage,
            }
        )
        return self._obs(), float(reward), terminated, truncated, info

    def _info(self) -> dict[str, float | int]:
        return {
            "remaining_capacity": self.remaining_capacity,
            "sold_units": self.total_sales,
            "total_revenue": self.total_revenue,
            "load_factor": self.total_sales / self.config.capacity,
            "period": self.t,
        }


class DynamicPricingEnv(_RevenueCore):
    """Discrete price-grid environment suitable for DQN."""

    def __init__(self, config: RevenueConfig | None = None):
        super().__init__(config=config)
        self.price_grid = np.linspace(
            self.config.min_price, self.config.max_price, self.config.price_levels
        )
        self.action_space = spaces.Discrete(self.config.price_levels)

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        return self._step_price(float(self.price_grid[int(action)]))


class ContinuousPricingEnv(_RevenueCore):
    """Continuous price environment suitable for PPO or SAC."""

    def __init__(self, config: RevenueConfig | None = None):
        super().__init__(config=config)
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def step(self, action: np.ndarray):
        if not self.action_space.contains(np.asarray(action, dtype=np.float32)):
            raise ValueError(f"Invalid action: {action}")
        normalized = float(np.asarray(action).reshape(-1)[0])
        price = self.config.min_price + ((normalized + 1.0) / 2.0) * (
            self.config.max_price - self.config.min_price
        )
        return self._step_price(price)
