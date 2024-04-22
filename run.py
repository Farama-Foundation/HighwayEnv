# from highway_planning.ipynb
#@title Imports for env, agent, and visualisation.
# Environment
import gymnasium as gym
import highway_env

# Agent
from rl_agents.agents.common.factory import agent_factory

# Visualisation
import sys
from tqdm.notebook import trange
sys.path.insert(0, './highway-env/scripts/')
from scripts.utils import record_videos, show_videos

# Make environment
env = gym.make("intersection-v0", render_mode="rgb_array")
#env = record_videos(env)

(obs, info), done = env.reset(), False

# Make agent
agent_config = {
    "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
    "env_preprocessors": [{"method":"simplify"}],
    "budget": 50,
    "gamma": 0.7,
}
agent = agent_factory(env, agent_config)

# Test BRTCalculator  
from highway_env.envs.common.safety import BRTCalculator

conservative_BRT = BRTCalculator(env, conservative = True)
print(f"Done with conservative BRT")

# Run episode
#for step in trange(env.unwrapped.config["duration"], desc="Running..."):
#    action = agent.act(obs)
#    obs, reward, done, truncated, info = env.step(action)
    
env.close()
#show_videos()