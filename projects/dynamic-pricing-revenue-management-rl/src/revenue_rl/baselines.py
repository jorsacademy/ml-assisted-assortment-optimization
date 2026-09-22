from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .env import RevenueConfig


@dataclass
class FixedPricePolicy:
    price: float = 180.0

    def __call__(self, obs: np.ndarray, config: RevenueConfig) -> float:
        return float(np.clip(self.price, config.min_price, config.max_price))


@dataclass
class TimeMarkdownPolicy:
    start_price: float = 220.0
    end_price: float = 140.0

    def __call__(self, obs: np.ndarray, config: RevenueConfig) -> float:
        progress = float(obs[2])
        price = self.start_price + progress * (self.end_price - self.start_price)
        return float(np.clip(price, config.min_price, config.max_price))


@dataclass
class CapacityProtectionPolicy:
    """Simple EMSR-inspired heuristic using capacity pacing.

    If sales are ahead of a linear sell-through target, raise price to protect
    capacity for later high-value demand. If behind target, lower price.
    """

    base_price: float = 180.0
    adjustment: float = 70.0

    def __call__(self, obs: np.ndarray, config: RevenueConfig) -> float:
        capacity_fraction, time_remaining, progress, _ = map(float, obs)
        sold_fraction = 1.0 - capacity_fraction
        target_sold = progress
        gap = sold_fraction - target_sold
        late_premium = 25.0 * progress
        price = self.base_price + self.adjustment * gap + late_premium
        if time_remaining < 0.15 and capacity_fraction > 0.25:
            price -= 35.0
        return float(np.clip(price, config.min_price, config.max_price))
