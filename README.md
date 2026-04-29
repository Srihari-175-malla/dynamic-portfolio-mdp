# Dynamic Portfolio Optimization using Markov Decision Process (MDP)

## Overview
This repository implements dynamic asset allocation and portfolio optimization framed as a Markov Decision Process (MDP). It combines Mean-Variance Quadratic Programming, Conditional Value at Risk (CVaR / Expected Shortfall) risk management, and MDP Value Iteration under dynamic market regimes (Bull, Neutral, Bear).

## Key Features
- **Mean-Variance Quadratic Optimization**: Computes optimal Markowitz efficient frontier weights.
- **Conditional Value at Risk (CVaR)**: Evaluates tail risk at configurable confidence intervals (e.g., 95%, 99%).
- **MDP Value Iteration**: Solves for optimal dynamic rebalancing policies across market regimes.

## Installation & Running Tests
```bash
python -m unittest discover -s tests
```
