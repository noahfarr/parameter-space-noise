from dataclasses import dataclass

import jax
import optax
import jax.numpy as jnp


@dataclass(frozen=True)
class NoiseState:
    param_std: float = 0.2
    target_action_std: float = 0.2
    adaptation_coefficient: float = 1.01


def adapt_noise_state(state: NoiseState, distance: float) -> NoiseState:
    param_std = (
        state.param_std / state.adaptation_coefficient
        if distance > state.target_action_std
        else state.param_std * state.adaptation_coefficient
    )
    return NoiseState(
        param_std=param_std,
        target_action_std=state.target_action_std,
        adaptation_coefficient=state.adaptation_coefficient,
    )


def perturb_actor(key, actor_state, noise_state: NoiseState):
    noise = optax.tree_utils.tree_random_like(
        key,
        actor_state.params,
        sampler=lambda key, shape, dtype: noise_state.param_std
        * jax.random.normal(key, shape, dtype),
    )
    perturbed_actor_params = jax.tree_util.tree_map(
        lambda x, y: x + y, actor_state.params, noise
    )
    perturbed_actor_state = actor_state.replace(params=perturbed_actor_params)
    return perturbed_actor_state


def ddpg_distance(actions, noisy_actions):
    distance = jnp.sqrt(jnp.mean((actions - noisy_actions) ** 2))
    return distance
