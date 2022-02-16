from assistive_gym.envs.object_handover import ObjectHandoverEnv
from .reaching_object import ReachingObjectEnv
from .agents import human
from .agents.stretch import Stretch
from .agents.human import Human

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

robot_arm = 'left'
human_controllable_joint_indices = human.right_arm_joints

class ReachingObjectStretchEnv(ReachingObjectEnv):
    def __init__(self):
        super(ReachingObjectStretchEnv,self).__init__(robot = Stretch('wheel_'+robot_arm), human = Human(human_controllable_joint_indices,controllable=False))
