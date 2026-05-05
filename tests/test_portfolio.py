import unittest
import sys, os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from portfolio_mdp import PortfolioMDPOptimizer

class TestPortfolioMDP(unittest.TestCase):
    def setUp(self):
        self.assets = ['Asset_A', 'Asset_B', 'Asset_C']
        self.mu = [0.12, 0.08, 0.04]
        self.cov = [
            [0.04, 0.01, 0.002],
            [0.01, 0.02, 0.001],
            [0.002, 0.001, 0.005]
        ]
        self.opt = PortfolioMDPOptimizer(self.assets, self.mu, self.cov)

    def test_optimal_weights_sum_to_one(self):
        weights = self.opt.optimize_weights_mean_variance()
        self.assertAlmostEqual(np.sum(weights), 1.0, places=4)
        self.assertTrue(np.all(weights >= -1e-5))

    def test_cvar_calculation(self):
        np.random.seed(42)
        returns_sample = np.random.multivariate_normal(self.mu, self.cov, size=1000)
        w = np.array([0.5, 0.3, 0.2])
        var, cvar = self.opt.calculate_cvar(w, returns_sample, alpha=0.95)
        self.assertLessEqual(var, cvar)

if __name__ == '__main__':
    unittest.main()
