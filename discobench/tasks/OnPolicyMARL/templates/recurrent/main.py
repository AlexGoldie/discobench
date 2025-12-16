import json
from networks import RecurrentModule, ActorCritic
import jax
import jax.numpy as jnp
from config import config
from jaxmarl.environments import spaces
from make_env import make_env
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
            init_hstate = RecurrentModule.initialize_carry(env.num_agents, config["GRU_HIDDEN_DIM"])

            def _cond_fn(runner_state):
                _, _, _, _, _, done = runner_state
                return jnp.logical_not(done)

            def _env_step(runner_state):
                rng, last_obs, h_state, env_state, total_reward, done = runner_state

                rng, rng_step = jax.random.split(rng)

                last_obs = jax.tree.map(lambda x: x[jnp.newaxis, :], last_obs)
                obs_batch = batchify(last_obs, env.agents, env.num_agents)
                if config.get("GET_AVAIL_ACTIONS", False):
                    avail_actions = env.get_avail_actions(env_state.env_state)
                    avail_actions = jax.tree.map(lambda x: x[jnp.newaxis, :], avail_actions)
                    avail_actions = batchify(avail_actions, env.agents, env.num_agents)
                    ac_in = (obs_batch[jnp.newaxis, :], jnp.zeros((1, env.num_agents)), avail_actions)
                else:
                    ac_in = (obs_batch[jnp.newaxis, :], jnp.zeros((1, env.num_agents)))

                next_h_state, pi, _ = network.apply(params, h_state, ac_in)

                if config.get("CONTINUOUS", False):
                    action = pi.loc
                else:
                    action = pi.mode()

                env_act = unbatchify(action, env.agents, 1, env.num_agents)
                env_act = jax.tree.map(lambda x: x[0].squeeze(), env_act)

                next_obs, next_env_state, reward, done, info = env.step(
                    rng_step, env_state, env_act,
                )
                if '__all__' in reward:
                    reward = reward['__all__']
                else:
                    reward = reward[env.agents[0]]

                total_reward = total_reward + reward
                done = done['__all__']
                return (rng, next_obs, next_h_state, next_env_state, total_reward, done)

            runner_state = (rng, obs, init_hstate, env_state, 0, False)
            runner_state = jax.lax.while_loop(
                _cond_fn,
                _env_step,
                runner_state,
            )
            return runner_state[-2]

        rngs = jax.random.split(rng, num_episodes)
        total_rewards = jax.vmap(single_episode)(rngs)
        return total_rewards.mean()

    return eval

if __name__ == "__main__":
    from functools import partial

    rng = jax.random.PRNGKey(42)
    n_seeds = 1
    if not config.get("DONT_VMAP", False):
        rngs = jax.random.split(rng, n_seeds)
        train = make_train(config)
        train_jit = jax.jit(partial(train))
        out = jax.vmap(train_jit)(rngs)
        params = out['runner_state'][0].params
    else:
        params_list = []
        for _ in range(n_seeds):
            rng, rng_seed = jax.random.split(rng)
            train = make_train(config)
            train_jit = jax.jit(partial(train))
            out = train_jit(rng_seed)
            params_list.append(out['runner_state'][0].params)
        params = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *params_list)

    # evaluate the policy
    eval = make_eval(config, 16)
    rng = jax.random.PRNGKey(30)
    eval_out = jax.vmap(jax.jit(eval), in_axes=(0, None))(params, rng)
    print(json.dumps({
        "eval_return": float(eval_out.mean()),
        "eval_return_std": float(eval_out.std()),
    }))
