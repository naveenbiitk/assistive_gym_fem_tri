from .testing import TestingEnv
from .agents import stretch
from .agents.stretch import Stretch
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

robot_arm = 'right'
class TestingStretchEnv(TestingEnv):
    def __init__(self):
        super(TestingStretchEnv, self).__init__(robot=Stretch('wheel_'+robot_arm))
