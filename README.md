# Parameter Space Noise for Exploration

A JAX/Flax implementation of [Parameter Space Noise for Exploration](https://arxiv.org/pdf/1706.01905) (Plappert et al., 2017) built on top of [CleanRL](https://github.com/vwxyzjn/cleanrl)'s DDPG algorithm.

Instead of adding noise to actions (as in standard DDPG), parameter space noise perturbs the policy network's weights directly, producing more consistent and state-dependent exploration.

## Overview

This project provides two DDPG implementations for continuous control tasks:

- **Standard DDPG** (`ddpg_continuous_action_jax.py`) — baseline with action-space exploration noise
- **DDPG + Parameter Space Noise** (`ddpg_continuous_action_param_noise_jax.py`) — exploration via adaptive parameter perturbation

### Key features

- Adaptive noise scaling that maintains a target action-space distance between perturbed and unperturbed policies
- LayerNorm in the parameter-noise variant (excluded from perturbation)
- Experiment tracking with Weights & Biases and TensorBoard

## Project Structure

```
parameter_space_noise/
├── ddpg_continuous_action_jax.py               # Standard DDPG baseline
├── ddpg_continuous_action_param_noise_jax.py   # DDPG with parameter space noise
└── parameter_noise_jax.py                      # Noise adaptation and perturbation utilities
```

## Installation

Requires Python 3.10+.

```bash
poetry install
```

You'll also need JAX installed with your preferred backend (CPU/GPU). See [JAX installation](https://jax.readthedocs.io/en/latest/installation.html).

## Usage

Run the parameter-noise variant:

```bash
python parameter_space_noise/ddpg_continuous_action_param_noise_jax.py --env-id HalfCheetah-v4
```

Run the standard DDPG baseline:

```bash
python parameter_space_noise/ddpg_continuous_action_jax.py --env-id Hopper-v4
```

All hyperparameters are configurable via CLI flags (powered by [tyro](https://github.com/brentyi/tyro)). Use `--help` to see available options.

## How It Works

1. **Perturb** — At the start of each episode, Gaussian noise scaled by `param_std` is added to the actor's parameters (excluding LayerNorm layers)
2. **Collect** — The perturbed actor interacts with the environment, collecting transitions
3. **Adapt** — Every `adaptation_frequency` steps, the distance between perturbed and unperturbed actions is measured. If the distance exceeds `target_action_std`, noise is decreased; otherwise it is increased
4. **Train** — Standard DDPG updates are applied to the actor and critic

## References

- Plappert, M., et al. "Parameter Space Noise for Exploration." arXiv:1706.01905, 2017.
- Huang, S., et al. "CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms." JMLR, 2022.

## License

MIT
