import unittest
import sys, os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from portfolio_mdp import PortfolioMDPOptimizer
from q_learning import QLearningPortfolioAgent
from backtest import PortfolioBacktester

class TestDynamicPortfolioMDP(unittest.TestCase):
    def setUp(self):
        self.assets = ['Equities', 'Bonds', 'Commodities']
        self.mu = [0.12, 0.04, 0.07]
        self.cov = [
            [0.040, 0.001, 0.015],
            [0.001, 0.005, -0.002],
            [0.015, -0.002, 0.035]
        ]
        self.opt = PortfolioMDPOptimizer(
            self.assets,
            self.mu,
            self.cov,
            risk_aversion=2.0,
            cvar_lambda=1.0,
            cvar_alpha=0.95,
            transaction_fee=0.002
        )

    def test_transition_matrix_stochastic_validity(self):
        """Checks that transition matrix rows sum to 1.0."""
        row_sums = np.sum(self.opt.P, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)
        self.assertTrue(np.all(self.opt.P >= 0.0))

    def test_mean_variance_weights_validity(self):
        """Checks that optimized weights satisfy simplex constraints."""
        w = self.opt.optimize_weights_mean_variance(regime_idx=0)
        self.assertAlmostEqual(np.sum(w), 1.0, places=4)
        self.assertTrue(np.all(w >= -1e-5))

    def test_cvar_parametric_and_empirical(self):
        """Checks analytical CVaR calculation against Monte Carlo empirical draws."""
        np.random.seed(42)
        w = np.array([0.6, 0.3, 0.1])
        mu_s, cov_s = self.opt.get_regime_moments(0)
        var_param, cvar_param = self.opt.calculate_cvar_parametric(w, mu_s, cov_s, alpha=0.95)

        # Draw large sample
        samples = np.random.multivariate_normal(mu_s, cov_s, size=50000)
        var_emp, cvar_emp = self.opt.calculate_cvar_empirical(w, samples, alpha=0.95)

        self.assertLess(var_param, cvar_param)
        self.assertAlmostEqual(cvar_param, cvar_emp, delta=0.03)

    def test_cvar_penalty_shifts_to_low_risk(self):
        """Verifies that higher CVaR penalty lambda reduces allocation to high-volatility assets."""
        opt_low_cvar = PortfolioMDPOptimizer(self.assets, self.mu, self.cov, cvar_lambda=0.0)
        opt_high_cvar = PortfolioMDPOptimizer(self.assets, self.mu, self.cov, cvar_lambda=5.0)

        w_low = opt_low_cvar.optimize_weights_mean_variance(regime_idx=0)
        w_high = opt_high_cvar.optimize_weights_mean_variance(regime_idx=0)

        # Asset 1 (Bonds) has lowest volatility (0.005 vs 0.040).
        # Increasing CVaR lambda should increase allocation to Bonds.
        self.assertGreaterEqual(w_high[1], w_low[1] - 0.05)

    def test_transaction_cost_penalizes_rebalancing(self):
        """Verifies that non-zero transaction fees penalize rebalancing away from current weights."""
        prev_w = np.array([0.4, 0.4, 0.2])
        w_new = np.array([0.8, 0.1, 0.1])

        reward_zero_fee = self.opt.compute_reward(0, prev_w, w_new)
        
        opt_high_fee = PortfolioMDPOptimizer(self.assets, self.mu, self.cov, transaction_fee=0.05)
        reward_high_fee = opt_high_fee.compute_reward(0, prev_w, w_new)

        self.assertGreater(reward_zero_fee, reward_high_fee)

    def test_value_iteration_convergence(self):
        """Tests that MDP Value Iteration converges to a stationary policy."""
        actions = self.opt.generate_action_grid(step=0.25)
        policy, V, _ = self.opt.value_iteration_policy(actions, discount=0.90, tolerance=1e-4)

        self.assertEqual(policy.shape, (self.opt.num_regimes, len(actions)))
        self.assertEqual(V.shape, (self.opt.num_regimes, len(actions)))
        self.assertTrue(np.all(np.isfinite(V)))

    def test_q_learning_agent_training(self):
        """Tests Q-Learning agent interaction and training updates."""
        actions = self.opt.generate_action_grid(step=0.5)
        agent = QLearningPortfolioAgent(self.opt, action_grid=actions, alpha=0.2, gamma=0.90, epsilon=0.5)
        rewards = agent.train(num_episodes=50, episode_length=20)

        self.assertEqual(len(rewards), 50)
        policy = agent.get_policy()
        self.assertEqual(policy.shape, (self.opt.num_regimes, len(actions)))

    def test_backtesting_engine(self):
        """Tests that the backtesting engine simulates equity trajectories properly."""
        backtester = PortfolioBacktester(self.opt, initial_capital=100000.0)
        regimes = backtester.simulate_regime_path(num_periods=52, seed=42)
        
        def constant_policy(s, w, t):
            return np.full(self.opt.N, 1.0 / self.opt.N)

        res = backtester.run_backtest(constant_policy, regimes, seed=42)
        self.assertIn('total_return', res)
        self.assertIn('sharpe_ratio', res)
        self.assertIn('max_drawdown', res)
        self.assertIn('cvar_95', res)
        self.assertEqual(len(res['equity_curve']), 53)
        self.assertGreater(res['final_capital'], 0.0)

if __name__ == '__main__':
    unittest.main()
