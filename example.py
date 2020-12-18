import gym
import highway_env
from matplotlib import pyplot as plt
import pprint
from rl_agents.agents.common.factory import agent_factory
from tqdm import trange

env = gym.make('highway-multi-agent-v0')
# pprint.pprint(env.config)




# Make environment
obs, done = env.reset(), False

# # Make agent
# agent_config = {
#     "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
#     "env_preprocessors": [{"method":"simplify"}],
#     "budget": 50,
#     "gamma": 0.7,
# }
# agent = agent_factory(env, agent_config)



# Instead of this we can add stable baselines as a "model" and train the model.
agent_config = {
    "__class__": "<class 'rl_agents.agents.deep_q_network.pytorch.DQNAgent'>",
    "model": {
        "type": "MultiLayerPerceptron", # CNN, EgoAttentionModel, etc etc..
        "layers": [256, 256]
    },
    # "double": False,
    "gamma": 0.85, #0.8 Discount factor
    "n_steps": 1, # If n_steps is greater than one, the batch will be composed of lists of successive transitions.
    "batch_size": 32, #32 Sample a batch of transitions
    "memory_capacity": 15000, # Used in replay memory default-10000, file had 15000
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




for step in trange(env.unwrapped.config["duration"], desc="Running..."):
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        env.reset()


# for _ in range(10):
#     action = env.action_type.actions_indexes["IDLE"]
#     obs, reward, done, info = env.step(action)
#     env.render()
# done = False
# while not done:
#     env.step(env.action_space.sample())
#     env.render()

# plt.imshow(env.render(mode="rgb_array"))
# plt.show()