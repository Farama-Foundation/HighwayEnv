import gym
import highway_env
from  gym.spaces.utils import flatdim
from gym.spaces import Box
import numpy as np
from rl_agents.agents.common.factory import agent_factory
from rl_agents.trainer.evaluation import Evaluation
from gym.wrappers import Monitor
from tqdm import trange

env = gym.make("highway-multi-agent-v0")

agent_config = {
    "__class__": "<class 'rl_agents.agents.deep_q_network.pytorch.DQNAgent'>",
    "model": {
        "type": "MultiLayerPerceptron",
        "layers": [256, 256]
    },
    # "double": False,
    "gamma": 0.75, #0.8
    "n_steps": 1,
    "batch_size": 32, #32
    "memory_capacity": 15000,
    "target_update": 50,
    "exploration": {
        "method": "EpsilonGreedy",
        "tau": 6000,
        "temperature": 1.0,
        "final_temperature": 0.05
    },
    "loss_function": "l2"
}
agent = agent_factory(env, agent_config)


obs, done = env.reset(), False

evaluation = Evaluation(env, agent, num_episodes=3000, display_env=False)

# Run episode
for step in trange(env.unwrapped.config["duration"], desc="Running..."):
    action = agent.act(obs)
    print(action)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        env.reset()


