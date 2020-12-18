import gym
import highway_env


from rl_agents.agents.common.factory import agent_factory
from matplotlib import pyplot as plt

# Visualisation
import sys
from tqdm.notebook import trange
sys.path.insert(0, './marl/highway-env/scripts/')
# from utils import record_videos, show_videos, capture_intermediate_frames
import pprint
# Make environment
env = gym.make("highway-v0")



env.configure({
    "controlled_vehicles": 2,
    "action": {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction"
            # "type": "ContinuousAction"
        }
    },
    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
            "type": "Kinematics"
        }
    },
    'screen_height': 150,
    'screen_width': 300
    ,"vehicles_count": 10

    ,'lanes_count': 4 
    # ,"absolute" : True,
    # 'duration': 50,
})

# env = record_videos(env)
obs, done = env.reset(), False
# capture_intermediate_frames(env)
# pprint.pprint(env.config)

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
print(env.observation_space)

# Run episode
# for step in trange(env.unwrapped.config["duration"], desc="Running..."):
#     action = agent.act(obs)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#         env.reset()
        # break
plt.imshow(env.render(mode="rgb_array"))

env.close()
