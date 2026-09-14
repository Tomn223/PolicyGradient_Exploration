import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os
import json

class ActorNetwork(tf.keras.Model):
    def __init__(self, action_dim, hidden_units):
        super(ActorNetwork, self).__init__()
        self.dense1 = layers.Dense(hidden_units, activation='tanh')
        self.dense2 = layers.Dense(hidden_units, activation='tanh')
        self.mu_layer = layers.Dense(action_dim, activation=None)
        # State-independent log std — standard PPO practice
        self.log_std = tf.Variable(
            tf.zeros(action_dim), 
            trainable=True, 
            name='log_std'
        )
    
    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        mu = self.mu_layer(x)
        std = tf.exp(self.log_std)
        return mu, std

class CriticNetwork(tf.keras.Model):
    def __init__(self, hidden_units):
        super(CriticNetwork, self).__init__()
        self.dense1 = layers.Dense(hidden_units, activation='tanh')
        self.dense2 = layers.Dense(hidden_units, activation='tanh')
        self.value_layer = layers.Dense(1, activation=None)
    
    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.value_layer(x)
    
class PPO:
    
    def __init__(
        self,
        env,
        gamma=0.99,
        lam=0.95,
        clip_epsilon=0.2,
        learning_rate=3e-4,
        entropy_coef=0.0,
        value_coef=0.5,
        num_timesteps=2048,
        update_epochs=10,
        num_minibatches=32,
        hidden_units=64,
        max_grad_norm=0.5,
        total_timesteps=1000000
    ):
        
        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.num_timesteps = num_timesteps
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.hidden_units = hidden_units
        self.max_grad_norm = max_grad_norm
        self.total_timesteps = total_timesteps
        
        action_dim = env.action_space.shape[0]
        
        self.actor = ActorNetwork(action_dim, hidden_units)
        self.critic = CriticNetwork(hidden_units)
        
        # Single optimizer for both networks
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, epsilon=1e-5)
        
    def collect_rollout(self):
        states = np.zeros((self.num_timesteps,) + self.env.observation_space.shape, dtype=np.float32)
        actions = np.zeros((self.num_timesteps,) + self.env.action_space.shape, dtype=np.float32)
        rewards = np.zeros(self.num_timesteps, dtype=np.float32)
        dones = np.zeros(self.num_timesteps, dtype=np.float32)
        values = np.zeros(self.num_timesteps, dtype=np.float32)
        log_probs = np.zeros(self.num_timesteps, dtype=np.float32)
        
        episode_reward = 0.0
        
        state, _ = self.env.reset()
        
        for t in range(self.num_timesteps):
            states[t] = state
            state_input = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
            
            mu, std = self.actor(state_input)
            value = self.critic(state_input)

            mu = mu.numpy()[0]
            std = std.numpy()[0]
            
            action = mu + std * np.random.normal(size=mu.shape)
            
            clipped_action = np.clip(action,
                                    self.env.action_space.low,
                                    self.env.action_space.high)
            
            log_prob = self._gaussian_log_prob(
                tf.constant(action.reshape(1,-1), dtype=tf.float32),
                tf.constant(mu.reshape(1,-1), dtype=tf.float32),
                tf.constant(std.reshape(1,-1), dtype=tf.float32)
            )
            
            next_state, reward, terminated, truncated, _ = self.env.step(clipped_action)
            done = terminated or truncated
            
            actions[t] = action
            rewards[t] = reward
            dones[t] = float(done)
            values[t] = value.numpy()[0,0]
            log_probs[t] = log_prob.numpy()
            
            episode_reward += reward
            
            if done:
                self.log_episode_rewards.append(episode_reward)
                episode_reward = 0.0
                state, _ = self.env.reset()
            else:
                state = next_state
                
        state_input = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        next_value = self.critic(state_input).numpy()[0, 0]
        
        advantages, returns = self.compute_gae(rewards, values, next_value, dones)
        
        return states, actions, log_probs, advantages, returns, values
    
    def _gaussian_log_prob(self, actions, mu, std):
        variance = tf.square(std)
        log_probs = -0.5 * (tf.square(actions-mu) / variance + 
                            tf.math.log(2.0 * np.pi * variance))
        return tf.reduce_sum(log_probs, axis=-1)
    
    def compute_gae(self, rewards, values, next_value, dones):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        
        for t in reversed(range(self.num_timesteps)):
            if t == self.num_timesteps - 1:
                next_val = next_value
            else:
                next_val = values[t+1]
                
            nonterminal = 1.0 - dones[t]
                
            # TD Error
            delta = rewards[t] + self.gamma * next_val * nonterminal - values[t]
            
            last_gae = delta + self.gamma * self.lam * nonterminal * last_gae
            advantages[t] = last_gae
                
        returns = advantages + values
        
        return advantages, returns
        
    @tf.function
    def train_step(self, states, actions, old_log_probs, advantages, returns):
        states = tf.cast(states, tf.float32)
        actions = tf.cast(actions, tf.float32)
        old_log_probs = tf.cast(old_log_probs, tf.float32)
        advantages = tf.cast(advantages, tf.float32)
        returns = tf.cast(returns, tf.float32)
        
        with tf.GradientTape() as tape:
            mu, std = self.actor(states)
            new_log_probs = self._gaussian_log_prob(actions, mu, std)
            
            values = tf.squeeze(self.critic(states), axis=-1)
            
            ratio = tf.exp(new_log_probs - old_log_probs)
            
            advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
            
            unclipped = ratio * advantages
            clipped = tf.clip_by_value(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped))
            
            critic_loss = tf.reduce_mean(tf.square(returns-values))
            
            entropy = tf.reduce_mean(
                tf.reduce_sum(
                    0.5 * tf.math.log(2.0 * np.pi * np.e * tf.square(std)),
                    axis=-1
                )
            )
            
            total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
        grads = tape.gradient(total_loss,
                              self.actor.trainable_variables + 
                              self.critic.trainable_variables)
        
        grads , _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.optimizer.apply_gradients(zip(grads,
                                           self.actor.trainable_variables + 
                                           self.critic.trainable_variables))
        
        return actor_loss, critic_loss, entropy
    
    def train(self):
        self.log_episode_rewards = []
        self.log_actor_losses = []
        self.log_critic_losses = []
        self.log_entropies = []
        
        timesteps_collected = 0
        rollout = 0
        # episode_reward = 0
        # current_episode_rewards = []
        
        while timesteps_collected < self.total_timesteps:
            
            states, actions, old_log_probs, advantages, returns, values = self.collect_rollout()
            rollout += 1
            timesteps_collected += self.num_timesteps
            
            minibatch_size = self.num_timesteps // self.num_minibatches
            
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_entropies = []
            
            for epoch in range(self.update_epochs):
                
                indices = np.random.permutation(self.num_timesteps)
                
                for start in range(0, self.num_timesteps, minibatch_size):
                    end = start + minibatch_size
                    mb_indices = indices[start:end]
                    
                    actor_loss, critic_loss, entropy = self.train_step(
                        states[mb_indices],
                        actions[mb_indices],
                        old_log_probs[mb_indices],
                        advantages[mb_indices],
                        returns[mb_indices]
                    )
                    
                    epoch_actor_losses.append(float(actor_loss))
                    epoch_critic_losses.append(float(critic_loss))
                    epoch_entropies.append(float(entropy))
                    
            self.log_actor_losses.append(np.mean(epoch_actor_losses))
            self.log_critic_losses.append(np.mean(epoch_critic_losses))
            self.log_entropies.append(np.mean(epoch_entropies))
            
            if rollout % 10 == 0:
                print(f"Timesteps: {timesteps_collected}/{self.total_timesteps} | "
                      f"Average Reward: {np.mean(self.log_episode_rewards[-self.num_timesteps:]):.4f} | "
                        f"Actor Loss: {np.mean(epoch_actor_losses):.4f} | "
                        f"Critic Loss: {np.mean(epoch_critic_losses):.4f}")
            
    def save(self, base_path):
        os.makedirs(os.path.join(base_path, 'models'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'curves'), exist_ok=True)
        
        self.actor.save_weights(os.path.join(base_path, 'models', 'actor.weights.h5'))
        self.critic.save_weights(os.path.join(base_path, 'models', 'critic.weights.h5'))
        
        logs = {
            'episode_rewards': np.array(self.log_episode_rewards),
            'actor_losses': np.array(self.log_actor_losses),
            'critic_losses': np.array(self.log_critic_losses),
            'entropies': np.array(self.log_entropies),
        }
        np.savez(os.path.join(base_path, 'logs', 'training_logs.npz'), **logs)
        
        run_info = {
            'algorithm': 'ppo',
            'config': {
                'gamma': self.gamma,
                'lam': self.lam,
                'clip_epsilon': self.clip_epsilon,
                'learning_rate': self.learning_rate,
                'entropy_coef': self.entropy_coef,
                'value_coef': self.value_coef,
                'num_timesteps': self.num_timesteps,
                'update_epochs': self.update_epochs,
                'num_minibatches': self.num_minibatches,
                'hidden_units': self.hidden_units,
                'max_grad_norm': self.max_grad_norm,
                'total_timesteps': self.total_timesteps
            }
        }
        with open(os.path.join(base_path, 'logs', 'run_info.json'), 'w') as f:
            json.dump(run_info, f, indent=2)
        
        print(f"Model and logs saved to {base_path}")