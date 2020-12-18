
​
# Environment
import gym
import highway_env
​
# Agent
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines import HER, SAC, PPO2
​
"""## Visualize a few episodes
​
We first define a simple helper function for visualization of episodes:
"""
​
# !pip install gym pyvirtualdisplay
# !apt-get install -y xvfb python-opengl ffmpeg
​
# from IPython import display as ipythondisplay
# from pyvirtualdisplay import Display
from gym.wrappers import Monitor
# from pathlib import Path
# import base64
​
display = Display(visible=0, size=(1400, 900))
display.start()
​
# def show_video():
#     html = []
#     for mp4 in Path("video").glob("*.mp4"):
#         video_b64 = base64.b64encode(mp4.read_bytes())
#         html.append('''<video alt="{}" autoplay 
#                       loop controls style="height: 400px;">
#                       <source src="data:video/mp4;base64,{}" type="video/mp4" />
#                  </video>'''.format(mp4, video_b64.decode('ascii')))
#     ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))
​
"""## Training"""
​
# env = gym.make("parking-v0")
# model = HER('MlpPolicy', env, SAC, n_sampled_goal=4,
#             goal_selection_strategy='future',
#             verbose=1, buffer_size=int(1e6),
#             learning_rate=1e-3,
#             gamma=0.9, batch_size=256,
#             policy_kwargs=dict(layers=[256, 256, 256]), tensorboard_log="./policy_log/")
# model.learn(int(5e4))
​
env = make_vec_env('highway-v0', n_envs=4)
model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./policy_log/")
model.learn(total_timesteps=5000)
model.save("ppo_highway")
​
# del model # remove to demonstrate saving and loading
# ​
# model = PPO2.load("ppo_highway")
# ​
# # Commented out IPython magic to ensure Python compatibility.
# # https://www.tensorflow.org/tensorboard/tensorboard_in_notebooks
# # %load_ext tensorboard
# # %tensorboard --logdir /content/policy_log
# ​
# """Test the policy"""
# ​
# env = gym.make("highway-v0")
# # env = Monitor(env, './video', force=True, video_callable=lambda episode: True)
# for episode in trange(1, desc="Test episodes"):
#     obs, done = env.reset(), False
#     env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame
#     while not done:
#         action, _ = model.predict(obs)
#         obs, reward, done, info = env.step(action)
# env.close()
# show_video()