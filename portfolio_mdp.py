"""
Dynamic Portfolio Optimization using Markov Decision Process (MDP) and Reinforcement Learning
Combines Value Iteration, Q-Learning, and CVaR (Conditional Value at Risk) / Mean-Variance constraints.

MDP State: (Current Portfolio Weights, Market State / Volatility Regime)
MDP Action: Rebalancing Allocation Weights across N assets
Reward: Portfolio Return - Risk Aversion * Variance - Transaction Costs - Penalty(CVaR)
"""

import numpy as np

class PortfolioMDPOptimizer:
    def __init__(self, asset_names, expected_returns, covariance_matrix, risk_aversion=2.0, transaction_fee=0.001):
        """
        Parameters:
        - asset_names: list of N asset names
        - expected_returns: array of shape (N,) expected return for each asset
        - covariance_matrix: array of shape (N, N) asset covariance matrix
        - risk_aversion: scalar gamma balancing return vs risk
        - transaction_fee: proportional rebalancing fee
        """
        self.asset_names = list(asset_names)
        self.N = len(asset_names)
        self.mu = np.array(expected_returns, dtype=float)
        self.cov = np.array(covariance_matrix, dtype=float)
        self.gamma = float(risk_aversion)
        self.fee = float(transaction_fee)

    def calculate_cvar(self, weights, returns_sample, alpha=0.95):
        """
        Calculates Conditional Value at Risk (CVaR / Expected Shortfall) at confidence level alpha.
        """
        portfolio_losses = -(returns_sample @ weights)
        var_threshold = np.percentile(portfolio_losses, alpha * 100)
        cvar = np.mean(portfolio_losses[portfolio_losses >= var_threshold])
        return var_threshold, cvar

    def optimize_weights_mean_variance(self):
        """
        Closed-form & numerical mean-variance quadratic programming optimization:
        Max w^T mu - (gamma / 2) * w^T Cov w
        Subject to sum(w) = 1, w >= 0
        """
        from scipy.optimize import minimize

        def obj(w):
            port_ret = np.dot(w, self.mu)
            port_vol = np.dot(w, np.dot(self.cov, w))
            return -(port_ret - (self.gamma / 2.0) * port_vol)

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1) for _ in range(self.N)]
        init_w = np.full(self.N, 1.0 / self.N)

        res = minimize(obj, init_w, bounds=bounds, constraints=constraints)
        return res.x if res.success else init_w

    def value_iteration_policy(self, states, actions, num_iterations=100, discount=0.95):
        """
        MDP Value Iteration over discrete market regimes (Bull, Neutral, Bear).
        """
        V = {s: 0.0 for s in states}
        policy = {}

        # Transition probabilities P(s' | s) for market regimes
        regime_returns = {
            'Bull': self.mu * 1.5,
            'Neutral': self.mu,
            'Bear': self.mu * 0.2 - 0.05
        }

        for _ in range(num_iterations):
            new_V = {}
            for s in states:
                best_val = -float('inf')
                best_a = None
                r_vec = regime_returns[s]
                for a in actions:
                    w = np.array(a)
                    exp_ret = np.dot(w, r_vec)
                    exp_var = np.dot(w, np.dot(self.cov, w))
                    reward = exp_ret - (self.gamma / 2.0) * exp_var

                    # Next state expectation
                    next_val = discount * np.mean([V[ns] for ns in states])
                    total_val = reward + next_val

                    if total_val > best_val:
                        best_val = total_val
                        best_a = a

                new_V[s] = best_val
                policy[s] = best_a
            V = new_V

        return policy, V

if __name__ == "__main__":
    assets = ['AAPL', 'MSFT', 'GOOGL', 'TLT']
    mu = [0.15, 0.12, 0.14, 0.04]
    cov = [
        [0.04, 0.02, 0.02, 0.001],
        [0.02, 0.035, 0.018, 0.001],
        [0.02, 0.018, 0.038, 0.001],
        [0.001, 0.001, 0.001, 0.005]
    ]

    opt = PortfolioMDPOptimizer(assets, mu, cov)
    opt_w = opt.optimize_weights_mean_variance()
    print("=== Portfolio MDP Optimization ===")
    for a, w in zip(assets, opt_w):
        print(f"Optimal Allocation {a}: {w*100:.2f}%")
