import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os


env = gym.make('HalfCheetah-v5')
# env_eval = gym.make('HalfCheetah-v5', render_mode="human")
# obs, _ = env.reset()
# env.render()

print(f"Observation space: {env.observation_space}")
print(f"Sample observation: {env.observation_space.sample()}")

print(f"Action space: {env.action_space}")
print(f"Sample action: {env.action_space.sample()}")

gamma = 0.99
learning_rate = 0.001
num_episodes = 1500
episodes_per_batch = 50
entropy_coef = 0.001

class PolicyNetwork(tf.keras.Model):
    def __init__(self, hidden_units=128):
        super(PolicyNetwork, self).__init__()
        self.dense1 = layers.Dense(hidden_units, activation='relu')
        self.dense2 = layers.Dense(hidden_units, activation='relu') # Added a 2nd hidden layer for complex locomotion
        self.mu_layer = tf.keras.layers.Dense(env.action_space.shape[0], activation=None)
        self.sigma_layer = tf.keras.layers.Dense(env.action_space.shape[0], activation='softplus')
        
    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        mu = self.mu_layer(x)
        # Add a tiny epsilon to sigma to prevent division by zero or NaN
        sigma = self.sigma_layer(x) + 1e-3
        
        return mu, sigma
    
policy = PolicyNetwork()
optimizer = tf.keras.optimizers.Adam(learning_rate)

def compute_returns(rewards, gamma):
    returns = np.zeros_like(rewards, dtype=np.float32)
    running_return = 0
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma*running_return
        returns[t] = running_return
    return returns

@tf.function
def train_step(states, raw_actions, returns):
    # Ensure inputs are explicitly float32
    states = tf.cast(states, tf.float32)
    raw_actions = tf.cast(raw_actions, tf.float32)
    returns = tf.cast(returns, tf.float32)
    
    with tf.GradientTape() as tape:
        # Get mean and standard deviation from the policy network
        mu, sigma = policy(states)
        
        # Calculate the log-probability under the standard Gaussian distribution
        variance = tf.square(sigma)
        
        # Explicitly cast 2 * pi to float32
        pi_const = tf.constant(2.0 * 3.14159265359, dtype=tf.float32)
        
        gaussian_log_probs = -0.5 * (tf.square(raw_actions - mu) / variance + 
                                     tf.math.log(pi_const * variance))
        
        # Apply the Squashing Correction (Change of Variables)
        # FIX: Ensure 1.0 and 1e-6 constants are float32
        tanh_u = tf.math.tanh(raw_actions)
        squash_correction = tf.math.log(tf.constant(1.0, dtype=tf.float32) - tf.square(tanh_u) + tf.constant(1e-6, dtype=tf.float32))
        
        # Corrected log prob for each dimension
        # Note: We subtract squash_correction based on the change of variables math
        corrected_log_probs = gaussian_log_probs - squash_correction
        
        # Sum across all action dimensions
        action_log_probs = tf.reduce_sum(corrected_log_probs, axis=-1)
        
         # --- Entropy bonus ---
        e_const = tf.constant(2.0 * 3.14159265359 * 2.71828182846, dtype=tf.float32)  # 2*pi*e
        gaussian_entropy = 0.5 * tf.math.log(e_const * variance)  # per-dimension
        total_entropy = tf.reduce_sum(gaussian_entropy, axis=-1)  # sum over action dims
        mean_entropy = tf.reduce_mean(total_entropy)  # average over batch

        # --- REINFORCE loss with entropy bonus ---
        pg_loss = -tf.reduce_mean(action_log_probs * returns)
        loss = pg_loss - entropy_coef * mean_entropy  # subtract because we want to *maximize* entropy
        
        
    # Optimize the policy weights
    grads = tape.gradient(loss, policy.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy.trainable_variables))
    
    return loss

for update in range(num_episodes // episodes_per_batch):
        
    batch_states, batch_raw_actions, batch_returns = [], [], []
    batch_episode_rewards = []  # just for logging

    for _ in range(episodes_per_batch):
    
        state, _ = env.reset()
        done = False
        states, raw_actions, rewards = [], [], []
        
        while not done:
            state_input = np.array(state, dtype = np.float32).reshape(1,-1)
            
            mu, sigma = policy(state_input)
            mu = mu.numpy()[0]
            sigma = sigma.numpy()[0]
            
            raw_action = mu + sigma * np.random.normal(size=mu.shape)
            
            env_action = np.tanh(raw_action)
            
            next_state, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated
            
            states.append(state_input[0])
            raw_actions.append(raw_action)
            rewards.append(reward)
            
            state = next_state
    
        returns = compute_returns(rewards, gamma)

        batch_states.append(np.vstack(states))
        batch_raw_actions.append(np.vstack(raw_actions))
        batch_returns.append(returns)
        batch_episode_rewards.append(sum(rewards))

    # Pool everything from all episodes_per_batch episodes together
    states_batch = np.vstack(batch_states)
    raw_actions_batch = np.vstack(batch_raw_actions)
    returns_batch = np.concatenate(batch_returns)
    
    # Normalize ACROSS the whole batch, not per-episode
    returns_batch = (returns_batch - np.mean(returns_batch)) / (np.std(returns_batch) + 1e-9)
    
    loss = train_step(states_batch, raw_actions_batch, returns_batch)
    
    if update % 2 == 0:
        avg_reward = np.mean(batch_episode_rewards)
        print(f"Update {update}, avg episode reward: {avg_reward:.2f}, loss: {loss:.4f}")
        
        
save_dir = "models"
os.makedirs(save_dir, exist_ok=True)
policy.save_weights(os.path.join(save_dir, f"halfcheetah_reinforce_entropy_batch{episodes_per_batch}.weights.h5"))
print(f"Saved policy weights to {save_dir}")
env.close()