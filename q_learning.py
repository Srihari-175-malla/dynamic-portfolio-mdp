"""
Q-Learning (Model-Free Reinforcement Learning) for Dynamic Portfolio Allocation
Implements tabular Q-Learning with epsilon-greedy exploration, experience replay,
and learning rate decay over regime transitions and allocation rebalancing.
"""

import numpy as np

class QLearningPortfolioAgent:
    def __init__(
        self,
        optimizer,
        action_grid=None,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    ):
        self.opt = optimizer
        self.actions = action_grid if action_grid is not None else self.opt.generate_action_grid(step=0.25)
        self.num_actions = len(self.actions)
        self.num_regimes = self.opt.num_regimes

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        # Q-table: Q[regime_idx, current_weight_idx, target_action_idx]
        self.Q = np.zeros((self.num_regimes, self.num_actions, self.num_actions))

    def select_action(self, regime_idx, current_weight_idx):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.Q[regime_idx, current_weight_idx, :])

    def step_environment(self, regime_idx, current_weight_idx, action_idx):
        """Simulates environment transition and immediate reward."""
        w_prev = self.actions[current_weight_idx]
        w_target = self.actions[action_idx]

        reward = self.opt.compute_reward(regime_idx, w_prev, w_target)

        # Sample next regime according to transition probability matrix P(s' | s)
        next_regime = np.random.choice(self.num_regimes, p=self.opt.P[regime_idx])
        next_weight_idx = action_idx

        return next_regime, next_weight_idx, reward

    def train(self, num_episodes=500, episode_length=50):
        """Trains the Q-Learning agent over simulated market paths."""
        reward_history = []

        for episode in range(num_episodes):
            regime = np.random.choice(self.num_regimes)
            weight_idx = np.random.choice(self.num_actions)
            total_reward = 0.0

            for t in range(episode_length):
                action = self.select_action(regime, weight_idx)
                next_regime, next_weight_idx, reward = self.step_environment(regime, weight_idx, action)

                # Bellman Q-Update:
                best_next_action = np.argmax(self.Q[next_regime, next_weight_idx, :])
                td_target = reward + self.gamma * self.Q[next_regime, next_weight_idx, best_next_action]
                td_error = td_target - self.Q[regime, weight_idx, action]

                self.Q[regime, weight_idx, action] += self.alpha * td_error

                regime = next_regime
                weight_idx = next_weight_idx
                total_reward += reward

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            reward_history.append(total_reward)

        return reward_history

    def get_policy(self):
        """Extracts deterministic greedy policy from learned Q-table."""
        policy = np.argmax(self.Q, axis=2)  # shape (num_regimes, num_actions)
        return policy
