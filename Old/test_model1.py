import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os

# class PolicyNetwork(tf.keras.Model):
#     def __init__(self, hidden_units=128):
#         super(PolicyNetwork, self).__init__()
#         self.dense1 = layers.Dense(hidden_units, activation='relu')
#         self.dense2 = layers.Dense(hidden_units, activation='relu') # Added a 2nd hidden layer for complex locomotion
#         self.mu_layer = tf.keras.layers.Dense(env.action_space.shape[0], activation=None)
#         self.sigma_layer = tf.keras.layers.Dense(env.action_space.shape[0], activation='softplus')
        
#     def call(self, state):
#         x = self.dense1(state)
#         x = self.dense2(x)
#         mu = self.mu_layer(x)
#         # Add a tiny epsilon to sigma to prevent division by zero or NaN
#         sigma = self.sigma_layer(x) + 1e-3
        
#         return mu, sigma

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
save_dir = 'Actor-Critic/models'
env = gym.make('HalfCheetah-v5', render_mode="human")
policy = ActorCritic()
dummy_input = np.zeros((1, env.observation_space.shape[0]), dtype=np.float32)
policy(dummy_input)  # builds the model so weights can be loaded into it
policy.load_weights(os.path.join(save_dir, "halfcheetah_actor_critic_200.weights.h5"))

eval_state, _ = env.reset()
eval_done = False
eval_reward = 0

while not eval_done:
    state_input = np.array(eval_state, dtype=np.float32).reshape(1, -1)
    mu, _, _ = policy(state_input)  # Exploit the learned center point (no sampling noise)
    eval_action = np.tanh(mu.numpy()[0])
    
    eval_state, reward, terminated, truncated, _ = env.step(eval_action)
    eval_done = terminated or truncated
    eval_reward += reward
print(f"Eval Total Reward: {eval_reward:.2f}\n")
env.close()