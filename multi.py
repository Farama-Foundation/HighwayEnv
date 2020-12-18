!pip install git+https://github.com/eleurent/highway-env#egg=highway-env
import gym
import highway_env

# Agent
!pip install git+https://github.com/eleurent/rl-agents#egg=rl-agents
from rl_agents.agents.common.factory import agent_factory

# Visualisation
import sys
from tqdm.notebook import trange
!git clone https://github.com/eleurent/highway-env.git
!pip install gym pyvirtualdisplay
!apt-get install -y xvfb python-opengl ffmpeg
sys.path.insert(0, './highway-env/scripts/')
from utils import record_videos, show_videos, capture_intermediate_frames