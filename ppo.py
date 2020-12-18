import gym
import highway_env
from  gym.spaces.utils import flatdim
from gym.spaces import Box
import numpy as np
from rl_agents.agents.common.factory import agent_factory

# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines import PPO2

# env = gym.make('CartPole-v1')
env = gym.make("highway-multi-agent-v0")

# env = gym.make("highway-v0")


# env.configure({
#     "controlled_vehicles": 2,
#     "action": {
#         "type": "MultiAgentAction",
#         "action_config": {
#             "type": "DiscreteMetaAction"
#             # "type": "ContinuousAction"
#         }
#     },
#     "observation": {
#         "type": "MultiAgentObservation",
#         "observation_config": {
#             "type": "Kinematics"
#         }
#     },
#     'screen_height': 150,
#     'screen_width': 300
#     ,"vehicles_count": 10

#     ,'lanes_count': 4 
#     # ,"absolute" : True,
#     # 'duration': 50,
# })
# 
# 
# 
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


print(type(env.observation_space))
obs, done = env.reset(), False
# print(obs)
print("main")
print(type(env.action_space))
action = agent.act(obs)
print("main2")
print(action)
Observation, reward, done, d = env.step(action)
# print("main3")

# print(Observation)

# env.step((0,1))



    # def observe(self) -> np.ndarray:
    #     ob=obs_type.observe()
    #     print("inside",type(ob))

    #     for i in range(self.agents_observation_types)-1:
    #         ob = np.vstack((ob,obs_type.observe()))

    #     return ob
# for i in obs[0]:
# 	for j in o
# 	print(i)
# 	
# print(np.asarray(obs[0], dtype=spaces.dtype).flatten())
# print(flatdim(obs[0]))
# print(npobs[0])
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

# model = PPO2(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=10000)






# # Create 4 artificial transitions per real transition
# n_sampled_goal = 4

# # SAC hyperparams:
# model = PPO2('MlpPolicy', env, SAC, n_sampled_goal=n_sampled_goal,
#             goal_selection_strategy='future',
#             verbose=1, buffer_size=int(1e6),
#             learning_rate=1e-3,
#             gamma=0.95, batch_size=256,
#             policy_kwargs=dict(layers=[256, 256, 256]))

# model.learn(int(2e5))
# model.save('her_sac_highway')

# # Load saved model
# model = HER.load('her_sac_highway', env=env)

# obs = env.reset()

# # Evaluate the agent
# episode_reward = 0
# for _ in range(100):
#   action, _ = model.predict(obs)
#   obs, reward, done, info = env.step(action)
#   env.render()
#   episode_reward += reward
#   if done or info.get('is_success', False):
#     print("Reward:", episode_reward, "Success?", info.get('is_success', False))
#     episode_reward = 0.0
#     obs = env.reset()