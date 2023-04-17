from .env import AssistiveEnv
from .agents.human import Human
from .agents.human import right_arm_joints, left_arm_joints, torso_joints, head_joints, all_joints
controllable_joints = right_arm_joints + left_arm_joints + torso_joints + head_joints
import pybullet as p


def configure_human(human):
    pass


class OptimizationEnv(AssistiveEnv):
    def __init__(self):
        self.human = Human(controllable_joint_indices=controllable_joints, controllable=True)
        super(OptimizationEnv, self).__init__(robot=None, human=self.human, task='pose_analysis')
        self.steps = 0

    def step(self, action):
        self.take_step(action, action_multiplier=1.)
        observation = self._get_obs()
        self.steps += 1
        done = True if self.steps > 200 else False
        reward = 0.
        info = dict()
        return observation, reward, done, info

    def _get_obs(self):
        return []

    def reset(self):
        super(OptimizationEnv, self).reset()
        self.build_assistive_env(fixed_human_base=False)

        # Human init
        configure_human(self.human)
        self.human.set_on_ground()

        p.setGravity(0, 0, 0)
        p.resetDebugVisualizerCamera(cameraDistance=1.10, cameraYaw=40, cameraPitch=-45,
                                     cameraTargetPosition=[-0.2, 0, 0.75], physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
        return self._get_obs()