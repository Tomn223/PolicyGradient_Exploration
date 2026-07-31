import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


env = gym.make('HalfCheetah-v5')
env_eval = gym.make('HalfCheetah-v5', render_mode="human")
# obs, _ = env.reset()
# env.render()

print(f"Observation space: {env.observation_space}")
print(f"Sample observation: {env.observation_space.sample()}")

print(f"Action space: {env.action_space}")
print(f"Sample action: {env.action_space.sample()}")

gamma = 0.99
learning_rate = 0.001
num_episodes = 1000
batch_size = 64

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

def train_step(states, raw_actions, returns):
    # Ensure inputs are explicitly float32
    states = tf.cast(states, tf.float32)
    raw_actions = tf.cast(raw_actions, tf.float32)
    returns = tf.cast(returns, tf.float32)
    
    with tf.GradientTape() as tape:
        # 1. Get mean and standard deviation from the policy network
        mu, sigma = policy(states)
        
        # 2. Calculate the log-probability under the standard Gaussian distribution
        variance = tf.square(sigma)
        
        # FIX: Explicitly cast 2 * pi to float32
        pi_const = tf.constant(2.0 * 3.14159265359, dtype=tf.float32)
        
        gaussian_log_probs = -0.5 * (tf.square(raw_actions - mu) / variance + 
                                     tf.math.log(pi_const * variance))
        
        # 3. Apply the Squashing Correction (Change of Variables)
        # FIX: Ensure 1.0 and 1e-6 constants are float32
        tanh_u = tf.math.tanh(raw_actions)
        squash_correction = tf.math.log(tf.constant(1.0, dtype=tf.float32) - tf.square(tanh_u) + tf.constant(1e-6, dtype=tf.float32))
        
        # Corrected log prob for each dimension
        # Note: We subtract squash_correction based on the change of variables math
        corrected_log_probs = gaussian_log_probs - squash_correction
        
        # 4. Sum across all action dimensions
        action_log_probs = tf.reduce_sum(corrected_log_probs, axis=-1)
        
        # 5. Calculate REINFORCE loss
        loss = -tf.reduce_mean(action_log_probs * returns)
        
    # 6. Optimize the policy weights
    grads = tape.gradient(loss, policy.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy.trainable_variables))
    
    return loss

for episode in range(num_episodes):
    
    if episode % 100 == 0:
        print(f"\n--- Visualizing Policy at Episode {episode} ---")
        eval_state, _ = env_eval.reset()
        eval_done = False
        eval_reward = 0
        
        while not eval_done:
            state_input = np.array(eval_state, dtype=np.float32).reshape(1, -1)
            mu, _ = policy(state_input)  # Exploit the learned center point (no sampling noise)
            eval_action = np.tanh(mu.numpy()[0])
            
            eval_state, reward, terminated, truncated, _ = env_eval.step(eval_action)
            eval_done = terminated or truncated
            eval_reward += reward
        print(f"Eval Total Reward: {eval_reward:.2f}\n")
        env_eval.close()
        
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
    returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-9)
    
    states_batch = np.vstack(states)
    raw_actions_batch = np.vstack(raw_actions)
    
    train_step(states_batch, raw_actions_batch, returns)
    
    if episode % 100 == 0:
        print(f"Episode {episode}/{num_episodes}")
        
env.close()