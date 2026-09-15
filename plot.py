import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os

def load_logs(result_dir):
    return np.load(f'{result_dir}/logs/training_logs.npz')

def load_run_info(result_dir):
    with open(f"{result_dir}/logs/run_info.json", "r") as file:
        return json.load(file)

def plot_reinforce_batch_rewards(result_dir, window=10):
    logs = load_logs(result_dir)
    run_info = load_run_info(result_dir)
    
    avg_rewards = logs['avg_rewards']
    
    smoothed = np.convolve(avg_rewards, 
                           np.ones(window)/window, 
                           mode='valid')
    
    episodes_per_batch = run_info["config"]["episodes_per_batch"]
    
    plt.figure(figsize=(10, 5))
    update_to_episode = np.arange(1, len(avg_rewards)+1) * episodes_per_batch
    plt.plot(update_to_episode, avg_rewards,
         linewidth=2.5,
         color='steelblue',
         label=f'Batch Average (Batch Size = {episodes_per_batch})')

    smoothed_x = np.arange(window//2, window//2 + len(smoothed)) * episodes_per_batch
    plt.plot(smoothed_x, smoothed,
         linewidth=1.5,
         alpha=0.7,
         color='orange',
         label=f'Smoothed (Window = {window * episodes_per_batch})')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('REINFORCE Batch Average Training Curve')
    plt.legend()
    plt.savefig(f'{result_dir}/curves/Batch_reward_curve.png', dpi=150)
    plt.show()

def plot_reinforce_raw_rewards(result_dir):
    logs = load_logs(result_dir)
    all_rewards = logs['all_rewards']
    
    plt.figure(figsize=(10, 5))
    plt.plot(all_rewards, linewidth=0.75, label='Raw')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('REINFORCE Raw Reward Training Curve')
    plt.legend()
    plt.savefig(f'{result_dir}/curves/raw_reward.png', dpi=150)
    plt.show()

def plot_reinforce_baseline(result_dir):
    logs = load_logs(result_dir)
    run_info = load_run_info(result_dir)
    
    if 'baselines' not in logs:
        print("No baseline data found")
        return
    
    episodes_per_batch = run_info["config"]["episodes_per_batch"]
    avg_rewards = logs['avg_rewards']
    baselines = logs['baselines']
    
    update_to_episode = np.arange(1, len(avg_rewards)+1) * episodes_per_batch
    
    plt.figure(figsize=(10, 5))
    
    plt.plot(update_to_episode, avg_rewards,
             linewidth=2.5,
             color='steelblue',
             label='Batch Average')
    
    plt.plot(update_to_episode, baselines,
             linewidth=2,
             color='orange',
             linestyle='--',
             label='Baseline')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('REINFORCE — Baseline vs Average Reward')
    plt.legend()
    plt.savefig(f'{result_dir}/curves/baseline_curve.png', dpi=150)
    plt.show()
    
def plot_actor_critic_rewards(result_dir, window=10):
    logs = load_logs(result_dir)
    run_info = load_run_info(result_dir)
    episode_rewards = logs['episode_rewards']
    
    smoothed = np.convolve(episode_rewards, 
                            np.ones(window)/window, 
                            mode='valid')
    
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards,
            linewidth=2.5,
            color='steelblue',
            label=f'Per Episode')

    smoothed_x = np.arange(window//2, window//2 + len(smoothed))
    plt.plot(smoothed_x, smoothed,
            linewidth=1.5,
            alpha=0.7,
            color='orange',
            label=f'Smoothed (Window = {window})')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Actor Critic Training Curve')
    plt.legend()
    plt.savefig(f'{result_dir}/curves/Batch_reward_curve.png', dpi=150)
    plt.show()
    
def plot_ppo_rewards(result_dir, window=10):
    logs = load_logs(result_dir)
    run_info = load_run_info(result_dir)
    episode_rewards = logs['episode_rewards']
    
    smoothed = np.convolve(episode_rewards, 
                            np.ones(window)/window, 
                            mode='valid')
    smoothed_x = np.arange(window//2, window//2 + len(smoothed))
    
    bigger_window = window * 5
    more_smoothed = np.convolve(episode_rewards, 
                            np.ones(bigger_window)/bigger_window, 
                            mode='valid')
    more_smoothed_x = np.arange(bigger_window//2, bigger_window//2 + len(more_smoothed))
    
    plt.figure(figsize=(10, 5))
    plt.plot(smoothed_x, smoothed,
            linewidth=2.5,
            color='steelblue',
            label=f'Per Episode')

    plt.plot(more_smoothed_x, more_smoothed,
            linewidth=1.5,
            alpha=0.7,
            color='orange',
            label=f'Smoothed (Window = {window})')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Actor Critic Training Curve')
    plt.legend()
    plt.savefig(f'{result_dir}/curves/Batch_reward_curve.png', dpi=150)
    plt.show()
    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', '-a', type=str, required=True)
    parser.add_argument('--logpath', '-l', type=str, required=True)
    args = parser.parse_args()
    
    result_dir = args.logpath
    run_info = load_run_info(result_dir)
    
    if args.algorithm == 'reinforce':
        plot_reinforce_batch_rewards(result_dir)
        plot_reinforce_raw_rewards(result_dir)
        if run_info["config"]["baseline_on"]:
            plot_reinforce_baseline(result_dir)
    elif args.algorithm == 'actor_critic':
        plot_actor_critic_rewards(result_dir)
    elif args.algorithm == 'ppo':
        plot_ppo_rewards(result_dir)
    
if __name__ == '__main__':
    main()