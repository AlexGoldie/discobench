import json
from pathlib import Path
from typing import Any, NamedTuple, Sequence

from networks import ActorCritic
import flax.linen as nn
import jax
import jax.numpy as jnp
from config import config
from jaxmarl.environments import spaces
from make_env import make_env
from networks import ActorCritic
from train import make_train, batchify, unbatchify
from config import config
from activation import activation


# Evaluation loop
def make_eval(config, num_episodes):
    env = make_env()
    max_dim = jnp.argmax(jnp.array([env.observation_space(a).shape[-1] for a in env.agents]))
    def get_action_dim(action_space):
        if isinstance(action_space, spaces.Discrete):
            return action_space.n
        elif isinstance(action_space, spaces.Box):
            return action_space.shape[0]
        else:
            raise ValueError(f"Unsupported action space type: {type(action_space)}")

    network = ActorCritic(
        get_action_dim(env.action_space(env.agents[0])),
        config=config,
        activation=activation,
    )

    def eval(params, rng):
        def single_episode(rng):
            rng, rng_reset = jax.random.split(rng)
            obs, env_state = env.reset(rng_reset)

            def _cond_fn(runner_state):
                _, _, _, _, done = runner_state
                return jnp.logical_not(done)

            def _env_step(runner_state):
                rng, last_obs, env_state, total_reward, done = runner_state

                rng, rng_step = jax.random.split(rng)

                last_obs = jax.tree.map(lambda x: x[jnp.newaxis, :], last_obs)
                obs_batch = batchify(last_obs, env.agents, env.num_agents)
                pi, _ = network.apply(params, obs_batch)
                if config.get("CONTINUOUS", False):
                    action = pi.loc
                else:
                    action = pi.mode()

                env_act = unbatchify(action, env.agents, 1, env.num_agents)
                env_act = jax.tree.map(lambda x: x[0], env_act)

                next_obs, next_state, reward, done, info = env.step(
                    rng_step, env_state, env_act,
                )
                if '__all__' in reward:
                    reward = reward['__all__']
                else:
                    _reward = 0
                    for agent in env.agents:
                        _reward = _reward + reward[agent]
                    reward = _reward

                total_reward = total_reward + reward
                done = done['__all__']
                return (rng, next_obs, next_state, total_reward, done)

            runner_state = (rng, obs, env_state, 0, False)
            runner_state = jax.lax.while_loop(
                _cond_fn,
                _env_step,
                runner_state,
            )
            return runner_state[3]

        rngs = jax.random.split(rng, num_episodes)
        total_rewards = jax.vmap(single_episode)(rngs)
        return total_rewards.mean()

    return eval

if __name__ == "__main__":
    from functools import partial

    rng = jax.random.PRNGKey(42)
    n_seeds = 8
    rngs = jax.random.split(rng, n_seeds)
    train = make_train(config)
    train_jit = jax.jit(partial(train))
    out = jax.vmap(train_jit)(rngs)

    # evaluate the policy
    eval = make_eval(config, 16)
    rng = jax.random.PRNGKey(30)
    eval_out = jax.vmap(jax.jit(eval), in_axes=(0, None))(out['runner_state'][0].params, rng)
    print(json.dumps({
        "eval_return": float(eval_out.mean()),
        "eval_return_std": float(eval_out.std()),
    }))
