from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .baselines import CapacityProtectionPolicy, FixedPricePolicy, TimeMarkdownPolicy
from .env import ContinuousPricingEnv, DynamicPricingEnv, RevenueConfig


@dataclass
class EpisodeResult:
    revenue: float
    load_factor: float
    sold_units: int
    remaining_capacity: int
    sellout_period: int | None


def _price_to_discrete_action(price: float, env: DynamicPricingEnv) -> int:
    return int(np.argmin(np.abs(env.price_grid - price)))


def run_baseline_episode(
    policy: Callable[[np.ndarray, RevenueConfig], float], seed: int
) -> EpisodeResult:
    env = DynamicPricingEnv()
    obs, _ = env.reset(seed=seed)
    terminated = truncated = False
    sellout_period = None
    info = {}
    while not (terminated or truncated):
        price = policy(obs, env.config)
        action = _price_to_discrete_action(price, env)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated and sellout_period is None:
            sellout_period = int(info["period"])
    return EpisodeResult(
        revenue=float(info["total_revenue"]),
        load_factor=float(info["load_factor"]),
        sold_units=int(info["sold_units"]),
        remaining_capacity=int(info["remaining_capacity"]),
        sellout_period=sellout_period,
    )


def evaluate_baseline(policy, episodes: int = 100, seed: int = 100) -> dict[str, float]:
    results = [run_baseline_episode(policy, seed + i) for i in range(episodes)]
    sellouts = [r.sellout_period for r in results if r.sellout_period is not None]
    return {
        "mean_revenue": float(np.mean([r.revenue for r in results])),
        "std_revenue": float(np.std([r.revenue for r in results])),
        "mean_load_factor": float(np.mean([r.load_factor for r in results])),
        "sellout_rate": float(np.mean([r.sellout_period is not None for r in results])),
        "mean_sellout_period": float(np.mean(sellouts)) if sellouts else float("nan"),
    }


def evaluate_rl(model_path: str, algorithm: str, episodes: int = 100, seed: int = 100):
    try:
        from stable_baselines3 import DQN, PPO
    except ImportError as exc:
        raise SystemExit("Install RL dependencies with: pip install -e '.[rl]'") from exc

    if algorithm == "dqn":
        env = DynamicPricingEnv()
        model = DQN.load(model_path)
    else:
        env = ContinuousPricingEnv()
        model = PPO.load(model_path)

    results = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        terminated = truncated = False
        sellout_period = None
        info = {}
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated and sellout_period is None:
                sellout_period = int(info["period"])
        results.append(
            EpisodeResult(
                revenue=float(info["total_revenue"]),
                load_factor=float(info["load_factor"]),
                sold_units=int(info["sold_units"]),
                remaining_capacity=int(info["remaining_capacity"]),
                sellout_period=sellout_period,
            )
        )
    return {
        "mean_revenue": float(np.mean([r.revenue for r in results])),
        "std_revenue": float(np.std([r.revenue for r in results])),
        "mean_load_factor": float(np.mean([r.load_factor for r in results])),
        "sellout_rate": float(np.mean([r.sellout_period is not None for r in results])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--model", default=None)
    parser.add_argument("--algorithm", choices=["dqn", "ppo"], default="dqn")
    args = parser.parse_args()

    baselines = {
        "fixed": FixedPricePolicy(),
        "markdown": TimeMarkdownPolicy(),
        "capacity_protection": CapacityProtectionPolicy(),
    }
    for name, policy in baselines.items():
        print(name, evaluate_baseline(policy, episodes=args.episodes))

    if args.model:
        print(args.algorithm, evaluate_rl(args.model, args.algorithm, episodes=args.episodes))


if __name__ == "__main__":
    main()
