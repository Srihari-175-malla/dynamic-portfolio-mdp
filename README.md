# Dynamic Portfolio Optimization via Markov Decision Process (MDP) & Reinforcement Learning

A comprehensive, mathematically rigorous quantitative portfolio allocation system that combines **Discrete-Event Regime-Switching Markov Decision Processes (MDP)**, **Conditional Value-at-Risk (CVaR / Expected Shortfall)** tail constraints, **Transaction Cost Penalization**, and **Model-Free Q-Learning**.

---

## Architecture Overview

```
                      +-----------------------------+
                      |   Market Regime Generator   |
                      |   P(s' | s) Transition Mat  |
                      +--------------+--------------+
                                     |
                 +-------------------+-------------------+
                 |                                       |
                 v                                       v
    +-------------------------+             +-------------------------+
    |  MDP Value Iteration    |             |   Q-Learning RL Agent   |
    |  Bellman Dynamic Prog   |             |   Model-Free Q-Table    |
    +------------+------------+             +------------+------------+
                 |                                       |
                 +-------------------+-------------------+
                                     |
                                     v
                       +---------------------------+
                       |   Reward & Loss Engine    |
                       |  • Mean Return E[R_p]     |
                       |  • Variance Penalty       |
                       |  • Analytical CVaR_alpha  |
                       |  • Turnover Transaction   |
                       +-------------+-------------+
                                     |
                                     v
                       +---------------------------+
                       |   Backtesting Simulator   |
                       |  • Multi-Period Trajectory|
                       |  • Sharpe Ratio & Max DD  |
                       |  • 95% Realized CVaR      |
                       +---------------------------+
```

---

## Mathematical Formulation

### 1. MDP State & Action Spaces
- **State Space**: $S = (s, \mathbf{w}_{\text{prev}})$, where $s \in \{\text{Bull}, \text{Neutral}, \text{Bear}\}$ is the discrete market regime and $\mathbf{w}_{\text{prev}}$ is the previous portfolio allocation vector.
- **Action Space**: $\mathcal{A} = \{ \mathbf{w} \in \mathbb{R}^N \mid \sum_{i=1}^N w_i = 1, w_i \ge 0 \}$ discretized on the unit simplex.
- **Transition Dynamics**: $P(s_{t+1} = s' \mid s_t = s)$ defined by a stationary Markov transition matrix $\mathbf{P} \in \mathbb{R}^{K \times K}$.

### 2. Penalized Reward Function
The instantaneous reward balances expected portfolio return against variance, extreme tail risk (CVaR), and turnover friction:
$$R(s, \mathbf{w}_{\text{prev}}, \mathbf{w}) = \mathbf{w}^T \boldsymbol{\mu}_s - \frac{\gamma}{2} \mathbf{w}^T \boldsymbol{\Sigma}_s \mathbf{w} - \lambda_{\text{CVaR}} \cdot \text{CVaR}_{\alpha}(\mathbf{w}; s) - c_{\text{fee}} \|\mathbf{w} - \mathbf{w}_{\text{prev}}\|_1$$

where:
- $\boldsymbol{\mu}_s, \boldsymbol{\Sigma}_s$: Regime-dependent expected return vector and covariance matrix.
- $\text{CVaR}_{\alpha}(\mathbf{w}; s) = -\mathbf{w}^T \boldsymbol{\mu}_s + \sigma_p(\mathbf{w}) \frac{\phi(\Phi^{-1}(\alpha))}{1 - \alpha}$: Gaussian Expected Shortfall at confidence level $\alpha = 0.95$.
- $c_{\text{fee}} \|\mathbf{w} - \mathbf{w}_{\text{prev}}\|_1$: Linear proportional transaction fee on allocation turnover.

### 3. Bellman Optimality Equation (Value Iteration)
$$V^*(s, \mathbf{w}) = \max_{\mathbf{w}' \in \mathcal{A}} \left[ R(s, \mathbf{w}, \mathbf{w}') + \beta \sum_{s' \in \mathcal{S}} P(s' \mid s) V^*(s', \mathbf{w}') \right]$$

---

## Core Components

1. **`portfolio_mdp.py`**:
   - `PortfolioMDPOptimizer`: Implements parametric & empirical CVaR, Mean-Variance quadratic programming, and MDP Value Iteration with full transition matrix integration.
2. **`q_learning.py`**:
   - `QLearningPortfolioAgent`: Implements tabular $\epsilon$-greedy Q-Learning with Bellman temporal difference updates over stochastic regime transitions.
3. **`backtest.py`**:
   - `PortfolioBacktester`: Multi-period sequential backtester evaluating dynamic MDP, Q-Learning, and static benchmark strategies (Annualized Return, Volatility, Sharpe Ratio, Max Drawdown, 95% CVaR, Turnover).
4. **`tests/test_portfolio.py`**:
   - Comprehensive unit test suite covering matrix stochasticity, simplex constraints, CVaR sensitivity, transaction costs, value iteration, and backtesting.

---

## Quick Start & Usage

### 1. Running Value Iteration & Optimization
```python
from portfolio_mdp import PortfolioMDPOptimizer

assets = ['US_Equities', 'Intl_Equities', 'Treasuries', 'Commodities']
mu = [0.12, 0.10, 0.04, 0.07]
cov = [
    [0.035, 0.025, 0.001, 0.015],
    [0.025, 0.040, 0.000, 0.018],
    [0.001, 0.000, 0.005, -0.002],
    [0.015, 0.018, -0.002, 0.045]
]

opt = PortfolioMDPOptimizer(assets, mu, cov, risk_aversion=2.5, cvar_lambda=1.0, transaction_fee=0.001)
actions = opt.generate_action_grid(step=0.25)
policy, V, _ = opt.value_iteration_policy(actions, discount=0.95)
```

### 2. Running Multi-Period Backtesting
```bash
python backtest.py
```

### 3. Running Unit Tests
```bash
python -m unittest discover -s tests -v
```

---

## Unit Test Coverage
- `test_transition_matrix_stochastic_validity` : Checks $\sum_j P_{ij} = 1.0$.
- `test_mean_variance_weights_validity` : Checks $\sum w_i = 1, w_i \ge 0$.
- `test_cvar_parametric_and_empirical` : Validates analytical Gaussian CVaR vs empirical Monte Carlo ($N=50,000$).
- `test_cvar_penalty_shifts_to_low_risk` : Confirms increasing $\lambda_{\text{CVaR}}$ shifts capital to lower-risk assets.
- `test_transaction_cost_penalizes_rebalancing` : Confirms non-zero transaction fees penalize turnover.
- `test_value_iteration_convergence` : Confirms Bellman residual convergence to stationary policy.
- `test_q_learning_agent_training` : Validates RL agent experience replay and Q-table updates.
- `test_backtesting_engine` : Tests dynamic simulation trajectories, drawdown calculation, and Sharpe metrics.

---

## License
MIT License
