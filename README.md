# TD3 Implementation for CS461 AI Course

This project implements the **Twin Delayed Deep Deterministic Policy Gradient (TD3)** algorithm as part of the CS461 Artificial Intelligence course programming assignment. TD3 is a state-of-the-art actor-critic method designed for continuous control tasks in reinforcement learning.

## Features
- **Actor-Critic Architecture:** Separate neural networks for the actor and two critics.
- **Target Networks:** Ensures stability by introducing delay in updates.
- **Policy Noise:** Adds noise to the target action to improve robustness.
- **Clipped Double Q-Learning:** Addresses overestimation bias by using the minimum of two Q-values.

## Requirements
- Python 3.8 or higher
- Required packages:
  - `numpy`
  - `torch`
  - `gym`
