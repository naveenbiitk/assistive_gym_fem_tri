from .show_phone import ShowPhoneEnv
from .agents import stretch, human
from .agents.stretch import Stretch
from .agents.pr2 import PR2
from .agents.human import Human

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

robot_arm='left'
human_controllable_joint_indices = human.right_arm_joints


class ShowPhoneStretchEnv(ShowPhoneEnv):
	def __init__(self):
		super(ShowPhoneStretchEnv, self).__init__(robot=Stretch(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))
		
class ShowPhoneStretchHumanEnv(ShowPhoneEnv, MultiAgentEnv):
    def __init__(self):
        super(ShowPhoneStretchHumanEnv, self).__init__(robot=Stretch(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ShowPhoneStretchHuman-v1', lambda config: ShowPhoneStretchHumanEnv())


class ShowPhonePR2Env(ShowPhoneEnv):
	def __init__(self):
		super(ShowPhonePR2Env, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))
		
class ShowPhonePR2HumanEnv(ShowPhoneEnv, MultiAgentEnv):
    def __init__(self):
        super(ShowPhonePR2HumanEnv, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ShowPhonePR2Human-v1', lambda config: ShowPhonePR2HumanEnv())

