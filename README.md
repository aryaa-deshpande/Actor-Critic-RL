# Actor-Critic Reinforcement Learning

Implementation of Actor-Critic methods including A3C (Asynchronous Advantage Actor-Critic) across multiple Gymnasium environments.

## Overview

This project was completed as part of CSE 546: Reinforcement Learning at the University at Buffalo (Spring 2026). It covers:

- **Part I:** Actor-Critic neural network architectures and training utilities
- **Part II:** Full A3C implementation with multi-threaded training on CartPole-v1
- **Part III:** A3C applied to LunarLander-v3 and Acrobot-v1

---

## Part I: Actor-Critic Foundations

### Architectures Implemented
- **Separate networks** - independent actor and critic with no shared layers
- **Shared network with two heads** - shared feature extractor with separate actor and critic output heads
- **Shared network for continuous actions** - outputs mean and log_std for Gaussian policy
- **CNN-based shared network** - for image-based environments like Atari

### Auto-Network Builder
`create_shared_network(env)` automatically builds the right architecture for any Gymnasium environment by inspecting the observation and action spaces.

Tested on:
- `CliffWalking-v0` - discrete observations
- `LunarLander-v3` - vector observations, discrete actions
- `PongNoFrameskip-v4` - image observations
- `HalfCheetah-v5` - continuous actions

### Utilities
- `normalize_observation(obs, env)` - normalizes Box observations to [-1, 1]
- Gradient clipping using `torch.nn.utils.clip_grad_norm_`

---

## Part II: A3C on CartPole-v1

### Implementation Details
- 2 parallel worker threads using Python `threading`
- Each thread maintains its own environment instance
- Local model copy per episode using `copy.deepcopy`
- Shared global model updated with `threading.Lock()` for safe synchronization
- Entropy bonus to encourage exploration
- Gradient clipping with `max_norm=0.5`

### Results
- Training: rewards consistently in 150-300 range by end of training
- Greedy evaluation (10 episodes): rewards between 159-191
- Video of greedy episode saved

---

## Part III: A3C on Complex Environments

### LunarLander-v3
- Hidden dim: 256, lr: 3e-4, entropy: 0.01, 1500 episodes
- Best eval rewards: -93 to -252, with one positive episode (+49)

### Acrobot-v1
- Hidden dim: 256, lr: 3e-4, entropy: 0.05, 2000 episodes
- Sparse reward environment - best eval achieved -146
- Results varied across runs due to stochastic nature of A3C



## Requirements

```bash
pip install gymnasium torch numpy matplotlib pandas
pip install "gymnasium[atari]" "autorom[accept-rom-license]"
pip install "gymnasium[mujoco]"
```

