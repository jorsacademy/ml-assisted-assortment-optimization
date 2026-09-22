from __future__ import annotations

import argparse
from pathlib import Path

from .env import ContinuousPricingEnv, DynamicPricingEnv


def train_dqn(total_timesteps: int, output: str, seed: int) -> None:
    try:
        from stable_baselines3 import DQN
    except ImportError as exc:
        raise SystemExit("Install RL dependencies with: pip install -e '.[rl]'") from exc

    env = DynamicPricingEnv()
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.25,
        exploration_final_eps=0.05,
        seed=seed,
        verbose=1,
    )
    model.learn(total_timesteps=total_timesteps)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    model.save(output)


def train_ppo(total_timesteps: int, output: str, seed: int) -> None:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit("Install RL dependencies with: pip install -e '.[rl]'") from exc

    env = ContinuousPricingEnv()
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        seed=seed,
        verbose=1,
    )
    model.learn(total_timesteps=total_timesteps)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    model.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["dqn", "ppo"], default="dqn")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output = args.output or f"models/{args.algorithm}_pricing"
    if args.algorithm == "dqn":
        train_dqn(args.timesteps, output, args.seed)
    else:
        train_ppo(args.timesteps, output, args.seed)


if __name__ == "__main__":
    main()
