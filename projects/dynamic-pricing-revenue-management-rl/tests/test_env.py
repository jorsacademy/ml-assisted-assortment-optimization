import numpy as np

from revenue_rl.env import ContinuousPricingEnv, DynamicPricingEnv, RevenueConfig


def test_discrete_env_reset_and_step_are_valid():
    env = DynamicPricingEnv()
    obs, info = env.reset(seed=123)
    assert env.observation_space.contains(obs)
    assert info["remaining_capacity"] == env.config.capacity

    obs, reward, terminated, truncated, info = env.step(0)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert info["price"] == env.price_grid[0]
    assert info["sales"] >= 0
    assert info["remaining_capacity"] <= env.config.capacity
    assert not (terminated and truncated)


def test_continuous_action_maps_to_price_bounds():
    env = ContinuousPricingEnv()
    env.reset(seed=1)
    _, _, _, _, low_info = env.step(np.array([-1.0], dtype=np.float32))
    assert low_info["price"] == env.config.min_price

    env.reset(seed=1)
    _, _, _, _, high_info = env.step(np.array([1.0], dtype=np.float32))
    assert high_info["price"] == env.config.max_price


def test_episode_never_sells_above_capacity():
    config = RevenueConfig(capacity=10, base_demand=50.0, horizon=5)
    env = DynamicPricingEnv(config)
    env.reset(seed=7)
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(0)
    assert info["sold_units"] <= config.capacity
    assert info["remaining_capacity"] >= 0
    assert 0.0 <= info["load_factor"] <= 1.0


def test_seed_reproducibility():
    env1 = DynamicPricingEnv()
    env2 = DynamicPricingEnv()
    env1.reset(seed=99)
    env2.reset(seed=99)
    trajectory1 = [env1.step(4)[4]["sales"] for _ in range(5)]
    trajectory2 = [env2.step(4)[4]["sales"] for _ in range(5)]
    assert trajectory1 == trajectory2
