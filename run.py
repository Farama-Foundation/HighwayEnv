# from highway_planning.ipynb
#@title Imports for env, agent, and visualisation.
# Environment
import gymnasium as gym
import highway_env
import numpy as np
from highway_env.envs.common.safety import SafetyWrapper

# Agent: commented for changes on my ec2 instance becasue I don't
# have the compute but will run locally 
from rl_agents.agents.common.factory import agent_factory

# Make environment
highway_env.register_highway_envs() #temp for local changes 
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

#conservative_BRT = BRTCalculator(env, conservative = True)
print(f"Done with conservative BRT")

duration = env.unwrapped.config["duration"]

old_results = np.load('./another_converged_brt.npy')

safe_env = SafetyWrapper(env, True, False, np.load('./new_converged_brt.npy'))
for i in range(duration):
    action = agent.act(obs)
    obs, reward, done, truncated, info, violation = safe_env.step(action)
    print(violation)
    safe_env.render()
    
safe_env.close()