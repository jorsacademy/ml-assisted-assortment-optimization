# Dynamic Pricing & Revenue Management with Reinforcement Learning

A compact industrial-engineering case study for capacity-constrained dynamic pricing under stochastic demand.

The project models a finite booking/selling horizon in which a seller controls price while a perishable capacity resource is consumed over time. The objective is not simply to maximize utilization; it is to maximize expected revenue while avoiding both premature sell-out and unsold capacity at the end of the horizon.

## Industrial context

Representative applications include:

- airline seats
- hotel rooms
- event tickets
- rental fleets
- cloud or service capacity
- appointment slots
- limited seasonal inventory

These problems are sequential: the price chosen now changes current demand and therefore the capacity available for future, potentially higher-value customers.

## MDP formulation

### State

The observation contains four normalized variables:

1. remaining capacity fraction
2. remaining time fraction
3. horizon progress
4. previous price, scaled to `[0, 1]`

### Actions

Two compatible environments are included.

`DynamicPricingEnv` uses a discrete price grid and is intended for DQN.

`ContinuousPricingEnv` exposes a normalized continuous action in `[-1, 1]`, mapped to the configured price range, and is intended for PPO or other continuous-control methods.

### Transition model

Customer arrivals follow a Poisson process whose mean depends on:

- current price
- time within the selling horizon
- a late-horizon high-value demand effect

Demand decreases exponentially with price and tends to strengthen later in the horizon.

### Reward

The immediate reward is period revenue. At the final period, unsold units receive a configurable spoilage penalty.

Conceptually:

`reward_t = price_t * sales_t - terminal_spoilage_penalty`

The environment enforces the physical capacity constraint, so cumulative sales can never exceed available inventory.

## Baselines

The project includes three interpretable reference policies.

### Fixed price

A single static price over the full horizon.

### Time markdown

Price declines linearly through time. This is a common operational heuristic when the dominant concern is clearing remaining capacity.

### Capacity protection

An EMSR-inspired pacing heuristic. It raises price when cumulative sales are ahead of a linear target and lowers price when inventory is being sold too slowly. It also includes a late-horizon clearance adjustment.

This is intentionally lightweight rather than a full classical EMSR implementation; its purpose is to provide an interpretable revenue-management baseline against which RL can be measured.

## Reinforcement-learning agents

### DQN

Use the discrete price-grid environment:

```bash
python -m revenue_rl.train --algorithm dqn --timesteps 50000
```

### PPO

Use the continuous-price environment:

```bash
python -m revenue_rl.train --algorithm ppo --timesteps 50000
```

Install RL dependencies first:

```bash
pip install -e '.[rl]'
```

## Evaluation

Evaluate the baseline policies:

```bash
python -m revenue_rl.evaluate --episodes 100
```

Evaluate a trained DQN policy as well:

```bash
python -m revenue_rl.evaluate --episodes 100 --algorithm dqn --model models/dqn_pricing
```

Or PPO:

```bash
python -m revenue_rl.evaluate --episodes 100 --algorithm ppo --model models/ppo_pricing
```

## Business KPIs

The evaluation layer reports metrics that matter in real revenue-management work:

- mean revenue
- revenue volatility
- load factor
- sell-out probability
- average sell-out period

A policy with a 100% load factor is not automatically good. Selling all units too early at low prices can generate less revenue than deliberately protecting capacity for later demand.

## Repository structure

```text
.
├── README.md
├── pyproject.toml
├── src/
│   └── revenue_rl/
│       ├── __init__.py
│       ├── env.py
│       ├── baselines.py
│       ├── train.py
│       └── evaluate.py
├── tests/
│   ├── test_env.py
│   └── test_baselines.py
└── .github/
    └── workflows/
        └── ci.yml
```

## Research extensions

Several useful academic extensions follow naturally from this base model:

- non-stationary or regime-switching demand
- customer segmentation and willingness-to-pay distributions
- cancellations and no-shows
- overbooking
- multi-product or network revenue management
- contextual pricing with demand forecasts
- distributional RL for tail-risk-aware pricing
- constrained RL for fairness or regulatory limits
- offline RL from historical transaction data
- benchmark comparison against approximate dynamic programming or stochastic programming

## Why this matters for industrial engineering

Revenue management combines operations research, stochastic modeling, forecasting, economics and sequential decision-making. The practical question is not “which RL algorithm is best?” but whether an adaptive pricing policy improves economic performance over transparent OR and heuristic baselines under realistic operational constraints.

## Tests

```bash
pip install -e '.[test]'
pytest -q
```

GitHub Actions runs the unit tests and a baseline-evaluation smoke test on Python 3.10, 3.11 and 3.12.
