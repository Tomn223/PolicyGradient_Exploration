import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os
import json
import argparse
from algorithms.reinforce import PolicyNetwork
from algorithms.ppo import ActorNetwork


parser = argparse.ArgumentParser()
parser.add_argument('-a', '--algorithm', type=str, required=True, 
                        choices=['reinforce', 'actor_critic', 'ppo'])
parser.add_argument('-t', '--timestamp', type=str, required=True)
args = parser.parse_args()

result_dir = os.path.join('results', args.algorithm, args.timestamp)
with open(f"{result_dir}/logs/run_info.json", "r") as file:
    run_info = json.load(file)

env = gym.make('HalfCheetah-v5', render_mode="human")

if args.algorithm == 'reinforce':
    policy = PolicyNetwork(env, run_info['config']['hidden_units'])
    modelName = f"reinforce{'_baseline' if run_info['config']['baseline'] else ''}_batch{run_info['config']['episodes_per_batch']}.weights.h5"
elif args.algorithm == 'actor_critic':
    pass
elif args.algorithm == 'ppo':
    policy = ActorNetwork(env.action_space.shape[0], run_info['config']['hidden_units'])
    modelName = "actor.weights.h5"
    
save_dir = os.path.join(result_dir, 'models')
dummy_input = np.zeros((1, env.observation_space.shape[0]), dtype=np.float32)
policy(dummy_input)  # builds the model so weights can be loaded into it
policy.load_weights(os.path.join(save_dir, modelName))

eval_state, _ = env.reset()
eval_done = False
eval_reward = 0

while not eval_done:
    state_input = np.array(eval_state, dtype=np.float32).reshape(1, -1)
    mu, _ = policy(state_input)  # Exploit the learned center point (no sampling noise)
    eval_action = np.tanh(mu.numpy()[0])
    
    eval_state, reward, terminated, truncated, _ = env.step(eval_action)
    eval_done = terminated or truncated
    eval_reward += reward
    
print(f"Eval Total Reward: {eval_reward:.2f}\n")
env.close()