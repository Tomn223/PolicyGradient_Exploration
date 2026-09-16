import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os
import json
import argparse
from algorithms.reinforce import PolicyNetwork
from algorithms.actor_critic import ActorNetwork


parser = argparse.ArgumentParser()
parser.add_argument('-a', '--algorithm', type=str, required=True, 
                        choices=['reinforce', 'actor_critic', 'ppo'])
parser.add_argument('-p', '--path', type=str, required=True)
parser.add_argument('-v', '--video', action='store_true')
parser.add_argument('-b', '--batch', type=int)
args = parser.parse_args()

result_dir = args.path
with open(f"{result_dir}/logs/run_info.json", "r") as file:
    run_info = json.load(file)

if args.video:
    env = gym.make('HalfCheetah-v5', render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=os.path.join(result_dir, 'videos'),
        episode_trigger=lambda episode_id: True
    )
elif args.batch:
    env = gym.make('HalfCheetah-v5')
else:
    env = gym.make('HalfCheetah-v5', render_mode="human")

if args.algorithm == 'reinforce':
    policy = PolicyNetwork(env, run_info['config']['hidden_units'])
    modelName = f"reinforce{'_baseline' if run_info['config']['baseline'] else ''}_batch{run_info['config']['episodes_per_batch']}.weights.h5"
elif args.algorithm == 'actor_critic' or args.algorithm == 'ppo':
    policy = ActorNetwork(env.action_space.shape[0], run_info['config']['hidden_units'])
    modelName = "actor.weights.h5"
    
save_dir = os.path.join(result_dir, 'models')
dummy_input = np.zeros((1, env.observation_space.shape[0]), dtype=np.float32)
policy(dummy_input)  # builds the model so weights can be loaded into it
policy.load_weights(os.path.join(save_dir, modelName))

num_runs = args.batch if args.batch else 1
eval_rewards = []

for i in range(num_runs):
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
        
    print(f"Eval {i+1} Total Reward: {eval_reward:.2f}\n")
    eval_rewards.append(eval_reward)
    
if args.batch:
    print(f"Mean Reward: {np.sum(eval_rewards) / num_runs:.2f}\n")
    print(f"Standard Deviation: {np.std(eval_rewards) :.2f}\n")
    
env.close()