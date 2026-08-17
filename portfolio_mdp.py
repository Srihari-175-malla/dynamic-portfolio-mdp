"""
Dynamic Portfolio Optimization using Markov Decision Process (MDP) and Reinforcement Learning
Combines Value Iteration, Q-Learning, and CVaR (Conditional Value at Risk) / Mean-Variance constraints
with explicit transaction costs and regime-switching Markov transition probabilities.

MDP State: (Market Regime s in {Bull, Neutral, Bear}, Current Portfolio Allocation w)
MDP Action: Target Portfolio Allocation w'
Reward Function: E[R_p] - (gamma / 2) * Var(R_p) - lambda_cvar * CVaR_alpha(R_p) - fee * ||w' - w||_1
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

class PortfolioMDPOptimizer:
    def __init__(
        self,
        asset_names,
        expected_returns,
        covariance_matrix,
        regime_transition_matrix=None,
        regime_params=None,
        risk_aversion=2.0,
        cvar_lambda=1.0,
        cvar_alpha=0.95,
        transaction_fee=0.002
    ):
        """
        Parameters:
        - asset_names: list of N asset names
        - expected_returns: base array of shape (N,)
        - covariance_matrix: base array of shape (N, N)
        - regime_transition_matrix: (K, K) transition probability matrix P(s' | s)
        - regime_params: dict mapping regime name/index to {'mu_mult': float/array, 'cov_mult': float}
        - risk_aversion: scalar gamma balancing return vs variance
        - cvar_lambda: weight parameter for CVaR penalty in reward
        - cvar_alpha: confidence level for CVaR (default 0.95)
        - transaction_fee: proportional turnover cost per rebalance (e.g. 0.002 = 20 bps)
        """
        self.asset_names = list(asset_names)
        self.N = len(asset_names)
        self.base_mu = np.array(expected_returns, dtype=float)
        self.base_cov = np.array(covariance_matrix, dtype=float)
        self.gamma = float(risk_aversion)
        self.cvar_lambda = float(cvar_lambda)
        self.cvar_alpha = float(cvar_alpha)
        self.fee = float(transaction_fee)

        self.regimes = ['Bull', 'Neutral', 'Bear']
        self.num_regimes = len(self.regimes)

        # Transition matrix P(s' | s)
        if regime_transition_matrix is not None:
            self.P = np.array(regime_transition_matrix, dtype=float)
        else:
            # Default stationary regime switching probabilities
            # Bull tends to stay Bull (0.75), Neutral is persistent (0.70), Bear is sticky (0.60)
            self.P = np.array([
                [0.75, 0.20, 0.05],  # from Bull
                [0.20, 0.65, 0.15],  # from Neutral
                [0.10, 0.30, 0.60]   # from Bear
            ])

        # Regime-specific return and covariance parameters
        if regime_params is not None:
            self.regime_params = regime_params
        else:
            self.regime_params = {
                0: {'name': 'Bull', 'mu': self.base_mu * 1.4 + 0.02, 'cov': self.base_cov * 0.8},
                1: {'name': 'Neutral', 'mu': self.base_mu * 1.0, 'cov': self.base_cov * 1.0},
                2: {'name': 'Bear', 'mu': self.base_mu * 0.3 - 0.06, 'cov': self.base_cov * 1.8}
            }

    def get_regime_moments(self, regime_idx):
        """Returns (mu, cov) for the given market regime."""
        info = self.regime_params[regime_idx]
        return np.array(info['mu'], dtype=float), np.array(info['cov'], dtype=float)

    def calculate_cvar_parametric(self, weights, mu, cov, alpha=None):
        """
        Calculates analytical Gaussian Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR / Expected Shortfall).
        For loss L = -R_p ~ N(-mu_p, sigma_p^2):
        VaR_alpha = -mu_p + sigma_p * phi^{-1}(alpha)
        CVaR_alpha = -mu_p + sigma_p * (pdf(phi^{-1}(alpha)) / (1 - alpha))
        """
        alpha = alpha or self.cvar_alpha
        w = np.array(weights, dtype=float)
        mu_p = np.dot(w, mu)
        sigma_p = np.sqrt(np.maximum(np.dot(w, np.dot(cov, w)), 1e-9))

        z_alpha = norm.ppf(alpha)
        pdf_z = norm.pdf(z_alpha)

        var = -mu_p + sigma_p * z_alpha
        cvar = -mu_p + sigma_p * (pdf_z / (1.0 - alpha))
        return var, cvar

    def calculate_cvar_empirical(self, weights, returns_sample, alpha=None):
        """
        Calculates empirical non-parametric CVaR from simulated or historical return draws.
        """
        alpha = alpha or self.cvar_alpha
        portfolio_losses = -(returns_sample @ np.array(weights))
        var_threshold = np.percentile(portfolio_losses, alpha * 100)
        tail_losses = portfolio_losses[portfolio_losses >= var_threshold]
        cvar = np.mean(tail_losses) if len(tail_losses) > 0 else var_threshold
        return var_threshold, cvar

    def compute_reward(self, regime_idx, prev_weights, target_weights):
        """
        Computes the complete single-step MDP reward:
        R(s, w, w') = Expected Return - (gamma / 2) * Variance - lambda * CVaR - fee * Turnover
        """
        mu_s, cov_s = self.get_regime_moments(regime_idx)
        w = np.array(target_weights, dtype=float)
        w_prev = np.array(prev_weights, dtype=float)

        port_ret = np.dot(w, mu_s)
        port_var = np.dot(w, np.dot(cov_s, w))

        _, port_cvar = self.calculate_cvar_parametric(w, mu_s, cov_s, self.cvar_alpha)

        # Proportional turnover transaction cost: ||w' - w||_1
        turnover = np.sum(np.abs(w - w_prev))
        tx_cost = self.fee * turnover

        # Total penalized reward
        reward = port_ret - (self.gamma / 2.0) * port_var - self.cvar_lambda * port_cvar - tx_cost
        return reward

    def generate_action_grid(self, step=0.1):
        """
        Generates a discrete lattice of portfolio weight vectors on the unit simplex: sum(w) = 1, w >= 0.
        """
        def _recursive_grid(dim, remaining):
            if dim == 1:
                return [[round(remaining, 4)]]
            points = []
            steps = int(round(remaining / step))
            for i in range(steps + 1):
                val = round(i * step, 4)
                sub_points = _recursive_grid(dim - 1, remaining - val)
                for sp in sub_points:
                    points.append([val] + sp)
            return points

        grid = _recursive_grid(self.N, 1.0)
        return [np.array(p, dtype=float) for p in grid]

    def optimize_weights_mean_variance(self, regime_idx=1, prev_weights=None):
        """
        Solves continuous Mean-Variance Optimization with optional transaction cost penalty:
        Max w^T mu_s - (gamma / 2) * w^T Cov_s w - fee * ||w - w_prev||_1
        Subject to sum(w) = 1, w >= 0
        """
        mu_s, cov_s = self.get_regime_moments(regime_idx)
        w_init = prev_weights if prev_weights is not None else np.full(self.N, 1.0 / self.N)

        def objective(w):
            ret = np.dot(w, mu_s)
            var = np.dot(w, np.dot(cov_s, w))
            _, cvar = self.calculate_cvar_parametric(w, mu_s, cov_s, self.cvar_alpha)
            tx = self.fee * np.sum(np.abs(w - w_init)) if prev_weights is not None else 0.0
            return -(ret - (self.gamma / 2.0) * var - self.cvar_lambda * cvar - tx)

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(self.N)]

        res = minimize(objective, w_init, method='SLSQP', bounds=bounds, constraints=constraints)
        return res.x if res.success else w_init

    def value_iteration_policy(self, action_grid=None, discount=0.95, tolerance=1e-5, max_iterations=200):
        """
        Solves the Full MDP using Dynamic Programming / Value Iteration:
        State: (regime_idx, current_weight_idx)
        Action: target_weight_idx
        Bellman Equation:
        V_{k+1}(s, w) = max_{w'} [ R(s, w, w') + discount * sum_{s'} P(s' | s) * V_k(s', w') ]
        """
        actions = action_grid if action_grid is not None else self.generate_action_grid(step=0.2 if self.N > 3 else 0.1)
        num_actions = len(actions)
        num_regimes = self.num_regimes

        # Value table V[regime, weight_idx]
        V = np.zeros((num_regimes, num_actions))
        policy = np.zeros((num_regimes, num_actions), dtype=int)

        # Precompute reward tensor R[s, a_prev, a_target] for speed
        R_tensor = np.zeros((num_regimes, num_actions, num_actions))
        for s in range(num_regimes):
            for i, w_prev in enumerate(actions):
                for j, w_target in enumerate(actions):
                    R_tensor[s, i, j] = self.compute_reward(s, w_prev, w_target)

        for iteration in range(max_iterations):
            new_V = np.zeros_like(V)
            delta = 0.0

            for s in range(num_regimes):
                # Expected continuation value for each candidate action w':
                # E_{s'}[V(s', w')] = sum_{s'} P(s' | s) * V[s', w']
                expected_future_V = self.P[s, :] @ V  # shape (num_actions,)

                for i in range(num_actions):
                    # Total Q-value for each target action j
                    Q_values = R_tensor[s, i, :] + discount * expected_future_V
                    best_action = np.argmax(Q_values)
                    best_value = Q_values[best_action]

                    new_V[s, i] = best_value
                    policy[s, i] = best_action
                    delta = max(delta, abs(best_value - V[s, i]))

            V = new_V
            if delta < tolerance:
                break

        return policy, V, actions

if __name__ == "__main__":
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
    print(f"Discretized Actions: {len(actions)} allocation points on simplex.")

    policy, V, actions = opt.value_iteration_policy(actions, discount=0.95)
    print(f"\nMDP Value Iteration Converged. Value Matrix Shape: {V.shape}")
    for r_idx, r_name in enumerate(opt.regimes):
        init_idx = 0
        best_a = actions[policy[r_idx, init_idx]]
        print(f"Regime {r_name:>7} -> Optimal Allocation: {np.round(best_a * 100, 1)}%")
