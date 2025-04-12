import os
import random
import time
from dataclasses import dataclass

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from parameter_noise_jax import (
    NoiseState,
    adapt_noise_state,
    perturb_actor,
    ddpg_distance,
)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 0
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "parameter-space-noise"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the environment id of the task"""
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    actor_learning_rate: float = 1e-4
    """the learning rate of the actor optimizer"""
    critic_learning_rate: float = 1e-3
    """the learning rate of the critic optimizer"""
    buffer_size: int = int(1e5)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1e-3
    """target smoothing coefficient"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    adaptation_frequency: int = 50
    """the frequency of adapting the perturbed actor"""
    weight_decay: float = 1e-2
    """the coefficient of the regularization term"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.NormalizeObservation(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        x = nn.Dense(64)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(64)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class Actor(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        x = nn.tanh(x)
        x = x * self.action_scale + self.action_bias
        return x


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            group="ddpg_continuous_action_param_noise_jax",
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, actor_key, perturbed_actor_key, adaptation_actor_key, qf1_key = (
        jax.random.split(key, 5)
    )

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])
    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device="cpu",
        handle_timeout_termination=False,
    )
    noise_state = NoiseState(param_std=0.2, target_action_std=0.2)

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)

    actor = Actor(
        action_dim=np.prod(envs.single_action_space.shape),
        action_scale=jnp.array((envs.action_space.high - envs.action_space.low) / 2.0),
        action_bias=jnp.array((envs.action_space.high + envs.action_space.low) / 2.0),
    )
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, obs),
        target_params=actor.init(actor_key, obs),
        tx=optax.adam(learning_rate=args.actor_learning_rate),
    )
    perturbed_actor_state = perturb_actor(perturbed_actor_key, actor_state, noise_state)

    qf = QNetwork()
    qf1_state = TrainState.create(
        apply_fn=qf.apply,
        params=qf.init(qf1_key, obs, envs.action_space.sample()),
        target_params=qf.init(qf1_key, obs, envs.action_space.sample()),
        tx=optax.adamw(
            learning_rate=args.critic_learning_rate, weight_decay=args.weight_decay
        ),
    )
    actor.apply = jax.jit(actor.apply)
    qf.apply = jax.jit(qf.apply)

    @jax.jit
    def update_critic(
        actor_state: TrainState,
        qf1_state: TrainState,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        terminations: np.ndarray,
    ):
        next_state_actions = (
            actor.apply(actor_state.target_params, next_observations)
        ).clip(
            -1, 1
        )  # TODO: proper clip
        qf1_next_target = qf.apply(
            qf1_state.target_params, next_observations, next_state_actions
        ).reshape(-1)
        next_q_value = (
            rewards + (1 - terminations) * args.gamma * (qf1_next_target)
        ).reshape(-1)

        def mse_loss(params):
            qf_a_values = qf.apply(params, observations, actions).squeeze()
            return ((qf_a_values - next_q_value) ** 2).mean(), qf_a_values.mean()

        (qf1_loss_value, qf1_a_values), grads1 = jax.value_and_grad(
            mse_loss, has_aux=True
        )(qf1_state.params)
        qf1_state = qf1_state.apply_gradients(grads=grads1)

        return qf1_state, qf1_loss_value, qf1_a_values

    @jax.jit
    def update_actor(
        actor_state: TrainState,
        qf1_state: TrainState,
        observations: np.ndarray,
    ):
        def actor_loss(params):
            return -qf.apply(
                qf1_state.params, observations, actor.apply(params, observations)
            ).mean()

        actor_loss_value, grads = jax.value_and_grad(actor_loss)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)
        actor_state = actor_state.replace(
            target_params=optax.incremental_update(
                actor_state.params, actor_state.target_params, args.tau
            )
        )

        qf1_state = qf1_state.replace(
            target_params=optax.incremental_update(
                qf1_state.params, qf1_state.target_params, args.tau
            )
        )
        return actor_state, qf1_state, actor_loss_value

    start_time = time.time()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        actions = actor.apply(perturbed_actor_state.params, obs)
        actions = np.array(actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            perturbed_actor_key, _ = jax.random.split(perturbed_actor_key)
            perturbed_actor_state = perturb_actor(
                perturbed_actor_key, actor_state, noise_state
            )

            for info in infos["final_info"]:
                print(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}"
                )
                writer.add_scalar(
                    "charts/episodic_return", info["episode"]["r"], global_step
                )
                writer.add_scalar(
                    "charts/episodic_length", info["episode"]["l"], global_step
                )
                break

        # TRY NOT TO MODIFY: save data to replay buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        data = rb.sample(args.batch_size)

        qf1_state, qf1_loss_value, qf1_a_values = update_critic(
            actor_state,
            qf1_state,
            data.observations.numpy(),
            data.actions.numpy(),
            data.next_observations.numpy(),
            data.rewards.flatten().numpy(),
            data.dones.flatten().numpy(),
        )
        actor_state, qf1_state, actor_loss_value = update_actor(
            actor_state,
            qf1_state,
            data.observations.numpy(),
        )

        if global_step % args.adaptation_frequency == 0:
            data = rb.sample(args.batch_size)

            adaptation_actor_key, _ = jax.random.split(adaptation_actor_key)
            adaptation_actor_state = perturb_actor(
                adaptation_actor_key, actor_state, noise_state
            )

            actions = actor.apply(actor_state.params, data.observations.numpy())
            noisy_actions = actor.apply(
                adaptation_actor_state.params, data.observations.numpy()
            )
            distance = ddpg_distance(actions, noisy_actions).item()
            noise_state = adapt_noise_state(noise_state, distance)

            writer.add_scalar("noise/param_std", noise_state.param_std, global_step)
            writer.add_scalar("noise/distance", distance, global_step)

        if global_step % 100 == 0:
            writer.add_scalar("losses/qf1_loss", qf1_loss_value.item(), global_step)
            writer.add_scalar("losses/qf1_values", qf1_a_values.item(), global_step)
            writer.add_scalar("losses/actor_loss", actor_loss_value.item(), global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar(
                "charts/SPS",
                int(global_step / (time.time() - start_time)),
                global_step,
            )

    envs.close()
    writer.close()
