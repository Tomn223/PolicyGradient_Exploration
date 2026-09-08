import collections
import gymnasium as gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm
import os

from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple

env = gym.make('HalfCheetah-v5')

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

class ActorCritic(tf.keras.Model):
    def __init__(self, hidden_units=128):
        super().__init__()
        self.common1 = layers.Dense(hidden_units, activation='relu')
        self.common2 = layers.Dense(hidden_units, activation='relu') # Added a 2nd hidden layer for complex locomotion
        self.actor_mu_layer = layers.Dense(env.action_space.shape[0], activation=None)
        self.actor_sigma_layer = layers.Dense(env.action_space.shape[0], activation='softplus')
        self.critic = layers.Dense(1)
        
    def call(self, state):
        x = self.common1(state)
        x = self.common2(x)
        actor_mu = self.actor_mu_layer(x)
        # Add a tiny epsilon to sigma to prevent division by zero or NaN
        actor_sigma = self.actor_sigma_layer(x) + 1e-3
        value = self.critic(x)
        
        return actor_mu, actor_sigma, value
    

# @tf.numpy_function(Tout=[tf.float32, tf.float32, tf.int32])
def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    return (state.astype(np.float32), np.array(reward, np.float32), np.array(done, np.int32))

def run_episode(initial_state, model, max_steps):
    
    action_log_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    
    state = initial_state
    
    for t in tf.range(max_steps):
        state_input = tf.reshape(tf.cast(state, tf.float32), [1, -1])
        
        mu, sigma, value = model(state_input)
        mu = tf.squeeze(mu, axis=0)
        sigma = tf.squeeze(sigma, axis=0)
        
        raw_action = mu + sigma * tf.random.normal(shape=tf.shape(mu))
        env_action = tf.math.tanh(raw_action)
        
        values = values.write(t, tf.squeeze(value))
        
        variance = tf.square(sigma)
        pi_const = tf.constant(2.0 * 3.14159265359, dtype=tf.float32)
        gaussian_log_prob = -0.5 * (tf.square(raw_action - mu) / variance + tf.math.log(pi_const * variance))
        # tanh_u = tf.math.tanh(raw_action)
        # squash_correction = tf.math.log(tf.constant(1.0, dtype=tf.float32) - tf.square(env_action) + tf.constant(1e-6, dtype=tf.float32))
        squash_correction = tf.math.log(1.0 - tf.square(env_action) + 1e-6)
        corrected_log_prob = gaussian_log_prob - squash_correction
        action_log_prob = tf.reduce_sum(corrected_log_prob)
        
        action_log_probs = action_log_probs.write(t, action_log_prob)
        
        state, reward, done = env_step(env_action.numpy())
        
        rewards = rewards.write(t, reward)
        
        if tf.cast(done, tf.bool):
            break
        
    action_log_probs = action_log_probs.stack()
    values = values.stack()
    rewards = rewards.stack()

    return action_log_probs, values, rewards

def get_expected_return(rewards, gamma):
    
    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)
    
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    # if standardize:
    #     returns = ((returns - tf.math.reduce_mean(returns)) /
    #             (tf.math.reduce_std(returns) + eps))

    return returns
    
    
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

def compute_loss(
    action_log_probs: tf.Tensor,
    values: tf.Tensor,
    returns: tf.Tensor) -> tf.Tensor:
    """Computes the combined Actor-Critic loss."""

    advantage = returns - tf.stop_gradient(values)

    advantage = (
        advantage - tf.reduce_mean(advantage)
    ) / (tf.math.reduce_std(advantage) + 1e-8)

    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

    critic_loss = huber_loss(values, returns)

    return actor_loss + critic_loss


# @tf.function
def train_step(
    initial_state: tf.Tensor,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    gamma: float,
    max_steps_per_episode: int) -> tf.Tensor:
  """Runs a model training step."""

  with tf.GradientTape() as tape:

    # Run the model for one episode to collect training data
    action_log_probs, values, rewards = run_episode(
        initial_state, model, max_steps_per_episode)

    # Calculate the expected returns
    returns = get_expected_return(rewards, gamma)

    # Convert training data to appropriate TF tensor shapes
    action_log_probs, values, returns = [
        tf.expand_dims(x, 1) for x in [action_log_probs, values, returns]]

    # Calculate the loss values to update our network
    loss = compute_loss(action_log_probs, values, returns)

  # Compute the gradients from the loss
  grads = tape.gradient(loss, model.trainable_variables)

  # Apply the gradients to the model's parameters
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  episode_reward = tf.math.reduce_sum(rewards)

  return episode_reward

model = ActorCritic()
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

min_episodes_criterion = 100
max_episodes = 1000
max_steps_per_episode = 5000

running_reward = 0

# The discount factor for future rewards
gamma = 0.99

# Keep the last episodes reward
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

t = tqdm.trange(max_episodes)
for i in t:
    initial_state, info = env.reset()
    initial_state = tf.constant(initial_state, dtype=tf.float32)
    episode_reward = int(train_step(
        initial_state, model, optimizer, gamma, max_steps_per_episode))

    episodes_reward.append(episode_reward)
    running_reward = statistics.mean(episodes_reward)

    t.set_postfix(
        episode_reward=episode_reward, running_reward=running_reward)

    # Show the average episode reward every 10 episodes
    if i != 0 and i % 10 == 0:
      print(f'Episode {i}: average reward: {statistics.mean(list(episodes_reward)[-10:])}')

    # if running_reward > reward_threshold and i >= min_episodes_criterion:
    #     break

# print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')

save_dir = "models"
os.makedirs(save_dir, exist_ok=True)
model.save_weights(os.path.join(save_dir, f"halfcheetah_actor_critic.weights.h5"))
print(f"Saved policy weights to {save_dir}")
env.close()