# Actor-Critic-RL


This project implements and explores Actor-Critic reinforcement learning methods, including both architectural design and full Advantage Actor-Critic (A2C/A3C) training. The goal is to build a generalizable RL framework that works across multiple Gymnasium environments with different observation and action spaces.

---

# Part I: Actor-Critic Foundations

This section focuses on understanding and implementing core components of Actor-Critic methods and building reusable neural architectures.

---

## 1. Actor-Critic Architectures

Two different Actor-Critic designs are implemented:

### (a) Separate Networks
- Independent actor network
- Independent critic network
- No shared parameters

**Why use this?**
- More flexibility in learning separate representations
- Useful when actor and critic require different feature extraction

---

### (b) Shared Network with Two Heads
- Shared feature extractor
- Two output heads:
  - Actor head (policy output)
  - Critic head (value estimation)

**Why use this?**
- More parameter efficient
- Shared learning improves feature reuse
- Typically more stable and faster to train

---

## 2. Generalized Actor-Critic Network Builder

A dynamic function `create_shared_network(env)` is implemented to automatically construct Actor-Critic models for any Gymnasium environment.

### Supported observation types:
- Discrete (converted to one-hot encoding)
- Box (vector inputs)
- Image-based inputs (with preprocessing support)

### Supported action spaces:
- Discrete actions
- Continuous actions (Box)
- Multi-discrete actions

---

### Tested Environments
- CliffWalking-v0  
- LunarLander-v3  
- PongNoFrameskip-v4 (with preprocessing wrappers)  
- HalfCheetah-v5  

The implementation is designed to generalize to unseen environments.

---

## 3. Observation Normalization

A normalization function is implemented:

```python
normalize_observation(obs, env)
````

### Behavior:

* Applies only to Box observation spaces with defined bounds
* Scales observations to range **[-1, 1]**
* Uses:

  * `env.observation_space.low`
  * `env.observation_space.high`

### Tested on:

* LunarLander-v3
* PongNoFrameskip-v4

---

## 4. Gradient Clipping

Gradient clipping is applied during training using PyTorch:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
```

### Purpose:

* Prevents exploding gradients
* Stabilizes training in deep reinforcement learning
* Improves convergence behavior

Gradient norms are logged before and after clipping to verify effectiveness.

---

# Part II: Advantage Actor-Critic (A2C/A3C)

This section implements a full Advantage Actor-Critic algorithm and evaluates it on a Gymnasium environment.

---

## Algorithm Overview

The implementation supports:

* A2C (synchronous updates)
* A3C-style multi-threaded training (2+ worker threads)

Each worker:

* Interacts with its own environment instance
* Computes gradients independently
* Updates a shared global model

This improves exploration diversity and stabilizes training.

---

## Training Setup

* Framework: PyTorch
* Environment: Gymnasium-compatible tasks
* Parallel workers: minimum 2 actor-learner threads
* Policy: stochastic during training, greedy during evaluation

---

## Evaluation Procedure

The trained agent is evaluated using a greedy policy:

* At least 10 episodes
* No exploration (deterministic action selection)
* Total reward per episode is recorded and plotted

---

## Rendering

A single greedy episode is rendered to verify learned behavior.

Outputs include:

* Step-by-step environment rendering
* Saved visual evidence (screenshots or video)

---

## Key Results

* Stable learning achieved using shared global updates
* Parallel workers improved exploration efficiency
* Agent successfully learned task-specific policy behavior
* Evaluation demonstrates consistent performance under greedy policy

---

# Requirements

* Python 3.10
* Gymnasium
* PyTorch
* NumPy
* Matplotlib

Install dependencies:

```bash
pip install gymnasium torch numpy matplotlib
```

---

# References

* Mnih, V. et al. (2016). *Asynchronous Methods for Deep Reinforcement Learning*. ICML. [https://arxiv.org/abs/1602.01783](https://arxiv.org/abs/1602.01783)
* Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
* Gymnasium Documentation: [https://gymnasium.farama.org](https://gymnasium.farama.org)
* PyTorch Documentation: [https://pytorch.org](https://pytorch.org)

---

# Notes

This project emphasizes:

* Generalizable RL architecture design
* Stability techniques (normalization + gradient clipping)
* Multi-environment compatibility
* Actor-Critic theory-to-practice implementation


