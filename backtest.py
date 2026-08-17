"""
Backtesting and Evaluation Engine for Dynamic Portfolio Optimization
Compares Dynamic MDP Value Iteration, Q-Learning RL, Static Markowitz, and Equal Weight 1/N
across simulated multi-period stochastic market regime paths.
"""

import numpy as np

class PortfolioBacktester:
    def __init__(self, optimizer, initial_capital=100000.0):
        self.opt = optimizer
        self.initial_capital = float(initial_capital)

    def simulate_regime_path(self, num_periods=252, initial_regime=0, seed=42):
        """Simulates a Markov chain trajectory of market regimes."""
        np.random.seed(seed)
        path = [initial_regime]
        for _ in range(1, num_periods):
            curr_s = path[-1]
            next_s = np.random.choice(self.opt.num_regimes, p=self.opt.P[curr_s])
            path.append(next_s)
        return path

    def run_backtest(self, policy_func, regime_path, seed=42):
        """
        Executes dynamic sequential backtesting tracking asset returns, rebalancing turnover, and net equity.
        """
        np.random.seed(seed)
        num_periods = len(regime_path)
        N = self.opt.N

        capital = self.initial_capital
        equity_curve = [capital]
        turnover_history = []
        realized_returns = []

        curr_weights = np.full(N, 1.0 / N)

        for t, s in enumerate(regime_path):
            target_weights = policy_func(s, curr_weights, t)

            # Calculate transaction fee
            turnover = np.sum(np.abs(target_weights - curr_weights))
            fee_cost = capital * self.opt.fee * turnover
            turnover_history.append(turnover)

            # Sample realized asset returns from regime distribution
            mu_s, cov_s = self.opt.get_regime_moments(s)
            asset_ret = np.random.multivariate_normal(mu_s / 52.0, cov_s / 52.0)  # weekly frequency

            # Portfolio net return
            port_ret = np.dot(target_weights, asset_ret)
            capital = (capital - fee_cost) * (1.0 + port_ret)

            realized_returns.append(port_ret)
            equity_curve.append(capital)
            curr_weights = target_weights

        realized_returns = np.array(realized_returns)
        equity_curve = np.array(equity_curve)

        # Performance Metrics
        total_return = (capital - self.initial_capital) / self.initial_capital
        mean_ret = np.mean(realized_returns) * 52.0
        vol_ret = np.std(realized_returns) * np.sqrt(52.0)
        sharpe = (mean_ret - 0.03) / vol_ret if vol_ret > 1e-6 else 0.0

        # Maximum Drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - peak) / peak
        max_dd = np.min(drawdowns)

        # Realized 95% CVaR
        _, cvar_95 = self.opt.calculate_cvar_empirical(np.ones(1), realized_returns[:, None], alpha=0.95)

        return {
            'final_capital': capital,
            'total_return': total_return,
            'annual_return': mean_ret,
            'annual_volatility': vol_ret,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'cvar_95': cvar_95,
            'avg_turnover': np.mean(turnover_history),
            'equity_curve': equity_curve
        }

if __name__ == "__main__":
    from portfolio_mdp import PortfolioMDPOptimizer
    from q_learning import QLearningPortfolioAgent

    assets = ['US_Stock', 'Intl_Stock', 'Bonds', 'Gold']
    mu = [0.12, 0.10, 0.04, 0.06]
    cov = [
        [0.035, 0.020, 0.001, 0.010],
        [0.020, 0.040, 0.000, 0.012],
        [0.001, 0.000, 0.004, -0.001],
        [0.010, 0.012, -0.001, 0.030]
    ]

    opt = PortfolioMDPOptimizer(assets, mu, cov, risk_aversion=2.5, cvar_lambda=1.5, transaction_fee=0.001)
    actions = opt.generate_action_grid(step=0.25)
    mdp_policy, _, _ = opt.value_iteration_policy(actions)

    backtester = PortfolioBacktester(opt)
    regimes = backtester.simulate_regime_path(num_periods=104, seed=123)

    # 1. Dynamic MDP Policy
    def policy_mdp(s, w_curr, t):
        # find nearest action idx for w_curr
        dists = [np.linalg.norm(w_curr - a) for a in actions]
        w_idx = np.argmin(dists)
        target_idx = mdp_policy[s, w_idx]
        return actions[target_idx]

    # 2. Equal Weight Buy & Hold
    def policy_eq(s, w_curr, t):
        return np.full(opt.N, 1.0 / opt.N)

    res_mdp = backtester.run_backtest(policy_mdp, regimes, seed=42)
    res_eq = backtester.run_backtest(policy_eq, regimes, seed=42)

    print("=== Multi-Period Backtest Comparison (2 Years) ===")
    print(f"Dynamic MDP -> Return: {res_mdp['total_return']*100:.2f}%, Sharpe: {res_mdp['sharpe_ratio']:.2f}, MaxDD: {res_mdp['max_drawdown']*100:.2f}%, CVaR(95%): {res_mdp['cvar_95']*100:.2f}%")
    print(f"Equal Weight -> Return: {res_eq['total_return']*100:.2f}%, Sharpe: {res_eq['sharpe_ratio']:.2f}, MaxDD: {res_eq['max_drawdown']*100:.2f}%, CVaR(95%): {res_eq['cvar_95']*100:.2f}%")
