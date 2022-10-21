import haiku as hk
import jax.numpy as jnp
import jax.random as rd
from jax import jit, value_and_grad as vgrad
import jax
import gym
import optax

from distrax import Normal

from src.buffer import Buffer


class PPO:
    def __init__(self, env, gamma=.99, eps=.1) -> None:
        self.env = gym.make(env, render_mode='rgb_array')

        low, high = self.env.action_space.low, self.env.action_space.high
        mean_scale = ((high + low) / 2, (high - low) / 2)
        (t_actor, t_critic) = self.build(self.env.action_space.shape[0], mean_scale) #TODO : adapt to different shapes, adapt to discret envs

        key = rd.PRNGKey(0)
        key1, key2 = rd.split(key)

        dummy_state = jnp.ones(self.env.observation_space.shape)

        self.actor_params = t_actor.init(key1, dummy_state)
        self._actor = hk.without_apply_rng(t_actor).apply

        self.critic_params = t_critic.init(key2, dummy_state)
        self._critic = hk.without_apply_rng(t_critic).apply
        
        self.optim = optax.adam(1e-4)
        self.critic_opt_state = self.optim.init(params=self.critic_params)
        self.actor_opt_state = self.optim.init(params=self.actor_params)

        self.buffer = Buffer()
        self.gamma = gamma
        self.eps = eps

    @staticmethod
    def build(action_dim, action_mean_scale):
        @hk.transform
        def actor(x):
            mean, scale = action_mean_scale
            init = hk.initializers.Constant(0)
            x = hk.Linear(128)(x)
            x = jax.nn.relu(x)
            mu = hk.Linear(action_dim, w_init=init, b_init=init)(x)
            mu = jax.nn.tanh(mu) * scale + mean
            std = hk.Linear(action_dim, w_init=init, b_init=init)(x)
            return mu, jnp.where(jnp.exp(std)<1., jnp.exp(std), 1.)
        
        @hk.transform
        def critic(x):
            init = hk.initializers.Constant(0)
            x = hk.Linear(128)(x)
            x = jax.nn.relu(x)
            x = hk.Linear(1, w_init=init, b_init=init)(x)
            return x
        return actor, critic

    def actor(self, state):
        return self._actor(self.actor_params, state)

    def critic(self, state):
        return self._critic(self.critic_params, state)
    
    def act(self, state):
        return self.actor(state)
        
    def rollout_one(self, state):
        mu, std = self.act(state)
        dist = Normal(mu, std)
        action, logprob = dist.sample_and_log_prob()
        next_state, reward, done, _, _ = self.env.step(action)
        return state, action, reward, done, next_state, logprob

    def rollout(self, steps):
        """
        Resets env then rollouts 'steps' steps, and stores them in the buffer
        """
        state = self.env.reset()
        for i in range(steps):
            transition = self.rollout_one(state)
            state = transition[0]
            self.buffer.update(*transition)
    
    def critic_loss(self, rewards, done, next_states, states):
            target_values = rewards + self.gamma * (1-done) * self.critic(next_states)
            critic_loss = .5 * jnp.mean((target_values - self.critic(states))**2)
            return critic_loss

    def update(self, key, training_iterations, batch_size):
        for i in range(training_iterations):
            batch = self.buffer.sample(key, batch_size)
            states = batch.states
            actions = batch.actions
            rewards = batch.rewards
            done = batch.done
            next_states = batch.next_states
            logprob = batch.logprob

            critic_loss, grads = vgrad(self.critic_loss)(
                rewards, done, next_states, states
            )
            updates, self.critic_opt_state = self.optim.update(grads, self.critic_opt_state)
            self.critic_params = optax.apply_updates(self.critic_params, updates)

            target_values = rewards + self.gamma * (1-done) * self.critic(next_states)
            advantage = jax.lax.stop_gradient(self.critic(states) - target_values)

            dist_new_actions = Normal(*self.actor(states))
            new_actions, new_logprob = dist_new_actions.sample_and_log_prob()

            ratio = jnp.exp(new_logprob - logprob)

            min1 = ratio * advantage
            min2 = jnp.clip(ratio, 1-self.eps, 1+self.eps)