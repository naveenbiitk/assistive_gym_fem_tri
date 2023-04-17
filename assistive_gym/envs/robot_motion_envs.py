from .robot_motion import RobotMotionEnv
from .agents import stretch, human
from .agents.stretch import Stretch
from .agents.pr2 import PR2
from .agents.human import Human

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

robot_arm='left'
#human_controllable_joint_indices = human.right_arm_joints
class RobotMotionStretchEnv(RobotMotionEnv):
	def __init__(self):
		super(RobotMotionStretchEnv, self).__init__(robot=Stretch(robot_arm), human=None)
		
