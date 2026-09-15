import collections
import gymnasium as gym
import numpy as np
# import statistics
import tensorflow as tf
import json
from matplotlib import pyplot as plt
from tensorflow.keras import layers
import os

class ActorCriticNetwork(tf.keras.Model):
    def __init__(self, action_dim, hidden_units=128):
        super().__init__()
        self.common1 = layers.Dense(hidden_units, activation='relu')
        self.common2 = layers.Dense(hidden_units, activation='relu')
        self.actor_mu_layer = layers.Dense(action_dim, activation=None)
        self.actor_sigma_layer = layers.Dense(action_dim, activation='softplus')
        self.critic = layers.Dense(1)
        
    def call(self, state):
        x = self.common1(state)
        x = self.common2(x)
        actor_mu = self.actor_mu_layer(x)
        actor_sigma = self.actor_sigma_layer(x) + 1e-3 # Add small num to sigma to prevent div by 0
        value = self.critic(x)
        
        return actor_mu, actor_sigma, value

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
    
class ActorCritic:
    
    def __init__(
        self,
        env,
        num_episodes=500,
        gamma=0.99,
        learning_rate=1e-4,
        critic_coef=0.5,
        hidden_units=128
    ):
        self.env = env
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.critic_coef = critic_coef
        self.hidden_units = hidden_units
        
        self.actor = ActorNetwork(env.action_space.shape[0], hidden_units)
        self.critic = CriticNetwork(hidden_units)

        # self.model = ActorCriticNetwork(env.action_space.shape[0], hidden_units)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.huber_loss = tf.keras.losses.Huber()
        
    
    
    def env_step(self, action: np.ndarray):
        state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return (state.astype(np.float32), np.array(reward, np.float32), np.array(done, np.int32))

    def train_step(self, state, raw_action, reward, next_state, done):
        state = tf.cast(state, tf.float32)
        raw_action = tf.cast(raw_action, tf.float32)
        next_state = tf.cast(next_state, tf.float32)
        
        with tf.GradientTape() as tape:
            mu, sigma = self.actor(state)
            value = self.critic(state)
            value = tf.squeeze(value)
            
            next_value = self.critic(next_state)
            next_value = tf.squeeze(next_value)
            
            td_target = tf.stop_gradient(
                reward + self.gamma * next_value * (1.0 - float(done))
            )
            td_error = td_target - value
            
            variance = tf.square(sigma)
            pi_const = tf.constant(2.0 * np.pi, dtype=tf.float32)
            gaussian_log_probs = -0.5 * (tf.square(raw_action - mu) / variance +
                                        tf.math.log(pi_const * variance))
            squash_correction = tf.math.log(
                tf.constant(1.0, dtype=tf.float32) - 
                tf.square(tf.math.tanh(raw_action)) + 
                tf.constant(1e-6, dtype=tf.float32)
            )
            corrected_log_probs = gaussian_log_probs - squash_correction
            action_log_prob = tf.reduce_sum(corrected_log_probs)
            
            actor_loss = -(action_log_prob * tf.stop_gradient(td_error))
            critic_loss = self.huber_loss(tf.reshape(td_target, [1]), tf.reshape(value, [1]))
            loss = actor_loss + self.critic_coef * critic_loss
        
        grads = tape.gradient(loss, self.actor.trainable_variables + self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables + self.critic.trainable_variables))
        
        return loss, actor_loss, critic_loss, tf.abs(td_error)

    def train(self):
        self.log_episode_rewards = []
        self.log_losses = []
        self.log_actor_losses = []
        self.log_critic_losses = []
        self.log_td_errors = []

        for i in range(self.num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_losses = []
            episode_actor_losses = []
            episode_critic_losses = []
            episode_td_errors = []

            while not done:
                state_input = np.array(state, dtype=np.float32).reshape(1, -1)
                
                # Sample action outside the tape
                mu, sigma = self.actor(state_input)
                mu = mu.numpy()[0]
                sigma = sigma.numpy()[0]
                raw_action = mu + sigma * np.random.normal(size=mu.shape)
                env_action = np.tanh(raw_action)
                
                next_state, reward, terminated, truncated, _ = self.env.step(env_action)
                done = terminated or truncated
                episode_reward += float(reward)
            
                loss, actor_loss, critic_loss, td_error = self.train_step(
                    state_input,
                    raw_action.reshape(1, -1),
                    np.float32(reward),
                    np.array(next_state, dtype=np.float32).reshape(1, -1),
                    done
                )
                
                episode_losses.append(float(loss))
                episode_actor_losses.append(float(actor_loss))
                episode_critic_losses.append(float(critic_loss))
                episode_td_errors.append(float(td_error))
                
                state = next_state

            self.log_episode_rewards.append(episode_reward)
            self.log_losses.append(np.mean(episode_losses))
            self.log_actor_losses.append(np.mean(episode_actor_losses))
            self.log_critic_losses.append(np.mean(episode_critic_losses))
            self.log_td_errors.append(np.mean(episode_td_errors))

            if (i + 1) % 5 == 0:
                print(f"Episode {i+1}: "
                    f"Avg Reward={np.mean(self.log_episode_rewards[-5:]):.2f}, "
                    f"Loss={np.mean(self.log_losses[-5:]):.4f}")
    
    def save(self, base_path):
        os.makedirs(os.path.join(base_path, 'models'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'curves'), exist_ok=True)
        
        # self.model.save_weights(
        #     os.path.join(base_path, 'models', 'actor_critic.weights.h5')
        # )
        
        self.actor.save_weights(os.path.join(base_path, 'models', 'actor.weights.h5'))
        self.critic.save_weights(os.path.join(base_path, 'models', 'critic.weights.h5'))
        
        logs = {
            'episode_rewards': np.array(self.log_episode_rewards),
            'losses': np.array(self.log_losses),
            'actor_losses': np.array(self.log_actor_losses),
            'critic_losses': np.array(self.log_critic_losses),
            'td_errors': np.array(self.log_td_errors),
        }
        np.savez(os.path.join(base_path, 'logs', 'training_logs.npz'), **logs)
        
        run_info = {
            'algorithm': 'actor_critic',
            'config': {
                'gamma': self.gamma,
                'learning_rate': self.learning_rate,
                'critic_coef': self.critic_coef,
                'num_episodes': self.num_episodes,
                'hidden_units': self.hidden_units,
            }
        }
        with open(os.path.join(base_path, 'logs', 'run_info.json'), 'w') as f:
            json.dump(run_info, f, indent=2)
        
        print(f"Model and logs saved to {base_path}")
