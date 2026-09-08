import argparse
import yaml
from algorithms.reinforce import REINFORCE
# from algorithms.actor_critic import ActorCritic
# from algorithms.ppo import PPO
import gymnasium as gym
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, required=True, 
                        choices=['reinforce', 'actor_critic', 'ppo'])
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--num_episodes', type=int, default=None)
    # parser.add_argument('--entropy', action='store_true')
    # parser.add_argument('--baseline', action='store_true')
    # parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    env = gym.make('HalfCheetah-v5')
    
    config_path = f'configs/{args.config}.yaml' if args.config else f'configs/{args.algorithm}.yaml'
    config = load_config(config_path)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if args.algorithm == 'reinforce':
        agent = REINFORCE(env, **config)
    # elif args.algorithm == 'actor_critic':
    #     agent = ActorCritic(env)
    # elif args.algorithm == 'ppo':
    #     agent = PPO(env)
    
    rewards = agent.train()
    agent.save(f'results/{args.algorithm}/{timestamp}')

if __name__ == '__main__':
    main()