import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os
import json

class PolicyNetwork(tf.keras.Model):
    def __init__(self, env, hidden_units):
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

class REINFORCE:
    
    def __init__(
            self,
            env,
            entropy=False,
            baseline=False,
            gamma=0.99,
            learning_rate=0.001,
            entropy_coef=0.001,
            baseline_momentum=0.9,
            num_episodes=1500,
            episodes_per_batch=50,
            hidden_units=128
            ):
        self.env = env
        self.hidden_units = hidden_units
        self.policy = PolicyNetwork(env, hidden_units)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.entropy = entropy
        self.baseline_on = baseline
        self.baseline = 0.0
        self.baseline_momentum = baseline_momentum if baseline else None
        self.entropy_coef = entropy_coef if entropy else 0.0
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.num_episodes=num_episodes
        self.episodes_per_batch = episodes_per_batch
        
    def compute_returns(self, rewards):
        returns = np.zeros_like(rewards, dtype=np.float32)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma*running_return
            returns[t] = running_return
        return returns
    
    @tf.function
    def train_step(self, states, raw_actions, advantages):
        # Ensure inputs are explicitly float32
        states = tf.cast(states, tf.float32)
        raw_actions = tf.cast(raw_actions, tf.float32)
        advantages = tf.cast(advantages, tf.float32)
        
        with tf.GradientTape() as tape:
            # 1. Get mean and standard deviation from the policy network
            mu, sigma = self.policy(states)
            
            # 2. Calculate the log-probability under the standard Gaussian distribution
            variance = tf.square(sigma)
            
            # Explicitly cast pi to float32
            two_pi_const = tf.constant(2.0 * 3.14159265359, dtype=tf.float32)
            
            gaussian_log_probs = -0.5 * (tf.square(raw_actions - mu) / variance + 
                                        tf.math.log(two_pi_const * variance))
            
            # 3. Apply the Squashing Correction (Change of Variables)
            tanh_u = tf.math.tanh(raw_actions)
            squash_correction = tf.math.log(tf.constant(1.0, dtype=tf.float32) - tf.square(tanh_u) + tf.constant(1e-6, dtype=tf.float32))
            corrected_log_probs = gaussian_log_probs - squash_correction
            
            # 4. Sum across all action dimensions
            action_log_probs = tf.reduce_sum(corrected_log_probs, axis=-1)
            
            # Entropy Bonus
            e_const = tf.constant(2.71828182846, dtype=tf.float32)
            gaussian_entropy = 0.5 * tf.math.log(two_pi_const * e_const * variance)
            total_entropy = tf.reduce_sum(gaussian_entropy, axis=-1)
            mean_entropy = tf.reduce_mean(total_entropy)
    
            # 5. Calculate REINFORCE loss
            pg_loss = -tf.reduce_mean(action_log_probs * advantages)
            loss = pg_loss - self.entropy_coef * mean_entropy  # entropy_coef is 0 if entropy is off, so no impact
            
        # 6. Optimize the policy weights
        grads = tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))
        
        return loss

    def train(self):
        
        self.log_avg_rewards = [] # average episode reward per update
        self.log_losses = [] # loss per update
        self.log_all_rewards = [] # every episode reward
        self.log_baselines = [] # baseline value per update (if on)
        
        for update in range(self.num_episodes // self.episodes_per_batch):
            batch_states, batch_raw_actions, batch_returns = [], [], []
            batch_episode_rewards = []

            for _ in range(self.episodes_per_batch):
            
                state, _ = self.env.reset()
                done = False
                states, raw_actions, rewards = [], [], []
                
                while not done:
                    state_input = np.array(state, dtype = np.float32).reshape(1,-1)
                    
                    mu, sigma = self.policy(state_input)
                    mu = mu.numpy()[0]
                    sigma = sigma.numpy()[0]
                    
                    raw_action = mu + sigma * np.random.normal(size=mu.shape)
                    
                    env_action = np.tanh(raw_action)
                    
                    next_state, reward, terminated, truncated, _ = self.env.step(env_action)
                    done = terminated or truncated
                    
                    states.append(state_input[0])
                    raw_actions.append(raw_action)
                    rewards.append(reward)
                    
                    state = next_state
            
                returns = self.compute_returns(rewards)

                batch_states.append(np.vstack(states))
                batch_raw_actions.append(np.vstack(raw_actions))
                batch_returns.append(returns)
                batch_episode_rewards.append(sum(rewards))

            # Pool everything from all episodes_per_batch episodes together
            states_batch = np.vstack(batch_states)
            raw_actions_batch = np.vstack(batch_raw_actions)
            returns_batch = np.concatenate(batch_returns)
            
            if self.baseline_on:
                self.baseline = self.baseline_momentum * self.baseline + (1 - self.baseline_momentum) * np.mean(returns_batch)
                advantages = returns_batch - self.baseline
            else:
                advantages = returns_batch
                
            loss = self.train_step(states_batch, raw_actions_batch, advantages)
            
            avg_reward = np.mean(batch_episode_rewards)
            
            self.log_avg_rewards.append(float(avg_reward))
            self.log_losses.append(float(loss.numpy()))
            self.log_all_rewards.extend(batch_episode_rewards)
            if self.baseline_on:
                self.log_baselines.append(float(self.baseline))
            
            if update % 2 == 0:
                print(f"Update {update}, avg episode reward: {avg_reward:.2f}, loss: {loss:.4f}")
        
    def save(self, base_path):
        os.makedirs(os.path.join(base_path, 'models'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'curves'), exist_ok=True)
        
        self.policy.save_weights(
            os.path.join(
                base_path,
                'models',
                f"reinforce{'_baseline' if self.baseline_on else ''}_batch{self.episodes_per_batch}.weights.h5"
            )
        )
        
        print(f"Model saved to {base_path}/models")
        
        logs = {
            'avg_rewards': np.array(self.log_avg_rewards),
            'losses': np.array(self.log_losses),
            'all_rewards': np.array(self.log_all_rewards),
        }
        if self.baseline_on:
            logs['baselines'] = np.array(self.log_baselines)
        
        np.savez(os.path.join(base_path, 'logs', 'training_logs.npz'), **logs)
        
        run_info = {
            'algorithm': 'reinforce',
            'config': {
                'baseline_on': self.baseline_on,
                'baseline': self.baseline,
                'baseline_momentum': self.baseline_momentum,
                'entropy': self.entropy,
                'gamma': self.gamma,
                'learning_rate': self.learning_rate,
                'episodes_per_batch': self.episodes_per_batch,
                'num_episodes': self.num_episodes,
                'hidden_units': self.hidden_units
            }
        }
        
        with open(f'{base_path}/logs/run_info.json', 'w') as f:
            json.dump(run_info, f, indent=2)