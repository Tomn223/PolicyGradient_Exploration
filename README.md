# Policy Gradient Exploration
Implementations of REINFORCE, Actor-Critic, and PPO with TensorFlow trained on HalfCheetah-v5

<p align="center">
  <br>
  <img src="results/ppo/20260914_105313/videos/rl-video-episode-0-sample.gif" width="600">
  <br>
  <em>Example evaluation of a model trained with ppo.py</em>
  <br>
</p>

## Summary of Results

## Project Structure
```
PolicyGradient_Exploration/
├── algorithms/
│   ├── reinforce.py
│   ├── actor_critic.py
│   └── ppo.py
├── configs/
│   ├── reinforce.yaml
│   ├── actor_critic.yaml
│   └── ppo.yaml
├── results/
├── plot.py
├── train.py
├── evaluate.py
└── README.md
```
## Installation
```bash
pip install -r requirements.txt
```
## Usage
Train an algorithm:
```bash
python train.py --algorithm reinforce
python train.py --algorithm reinforce --config configs/reinforce_custom.yaml
```

Evaluate a trained agent:
```bash
python evaluate.py --algorithm actor_critic --path results/actor_critic/20240906_143022
```

Generate training curves:
```bash
python plot.py --algorithm ppo --path results/ppo/20240906_143022
```

Note: I chose to read in the full relative path as it's easy to copy within VS Code
