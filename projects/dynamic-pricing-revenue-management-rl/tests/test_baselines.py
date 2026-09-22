from revenue_rl.baselines import CapacityProtectionPolicy, FixedPricePolicy, TimeMarkdownPolicy
from revenue_rl.env import RevenueConfig
from revenue_rl.evaluate import evaluate_baseline


def test_baseline_prices_within_bounds():
    config = RevenueConfig()
    obs = [0.5, 0.5, 0.5, 0.5]
    for policy in (FixedPricePolicy(), TimeMarkdownPolicy(), CapacityProtectionPolicy()):
        price = policy(obs, config)
        assert config.min_price <= price <= config.max_price


def test_evaluation_returns_business_kpis():
    metrics = evaluate_baseline(FixedPricePolicy(), episodes=3, seed=10)
    assert metrics["mean_revenue"] >= 0.0
    assert 0.0 <= metrics["mean_load_factor"] <= 1.0
    assert 0.0 <= metrics["sellout_rate"] <= 1.0
