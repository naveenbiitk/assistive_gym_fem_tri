from platform import java_ver
from turtle import down, left
import numpy as np
from assistive_gym.envs.object_handover import ObjectHandoverEnv
import pybullet as p
import os

from .env import AssistiveEnv
from .agents.furniture import Furniture
from .agents.agent import Agent

class ReachingObjectEnv(AssistiveEnv):

    def __init__(self, robot, human):
        super(ReachingObjectEnv,self).__init__(robot=robot, human=human, task='reaching_object', obs_robot_len = (23 + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices))), obs_human_len = (24 + len(human.controllable_joint_indices)))

    def take_step(self):
        input_keys = p.getKeyboardEvents(self.id)
        action = self._get_action(input_keys)
        self.take_step(action)

    def _get_action(self, input_keys):
        # actions:
        # w - move base forward
        # s - move base backward
        # a - rotate base left
        # d - rotate base right
        # 
        # h - move arm up
        # j - move arm down
        # k - extend arm
        # l - detract arm
        # 
        # u - open gripper
        # i - close gripper
        # o - tilt gripper left
        # p - tilt gripper right

        action_dict = {
        ord('w'): "move_base_forward",
        ord('s'): "move_base_backward",
        ord('a'): "rotate_base_left",
        ord('d'): "rotate_base_right",
        ord('h'): "move_arm_up",
        ord('j'): "move_arm_down",
        ord('k'): "extend_arm_out",
        ord('l'): "detract_arm_in",
        ord('o'): "rotate_gripper_left",
        ord('p'): "rotate_gripper_right",
        ord('u'): "open_gripper",
        ord('i'): "close_gripper",
        }

        action = None

        return action

    def reset(self):
        super(ReachingObjectEnv,self).reset()

        self.build_assistive_env('wheelchair')

        self.table = Furniture()
        self.table.init("table",self.directory,self.id,self.np_random)
        self.table.set_base_pos_orient([1.0, -0.5, 0.0],[0, 0, np.pi/2, np.pi/2])

        self.object = Agent()
        self.object.init(p.loadURDF(os.path.join(self.directory, 'handover_object', 'handover_object.urdf')),self.id,self.np_random, indices=-1)
        self.object.set_base_pos_orient([0.7, -1.1, 0.9],[0, 0, 0, 1])

        joints_positions = [(self.human.j_right_elbow, -90), 
                            (self.human.j_left_elbow, -90), 
                            (self.human.j_right_hip_x, -90), 
                            (self.human.j_right_knee, 80), 
                            (self.human.j_left_hip_x, -90), 
                            (self.human.j_left_knee, 80), 
                            (self.human.j_head_x, self.np_random.uniform(-30, 30)), 
                            (self.human.j_head_y, self.np_random.uniform(-30, 30)), 
                            (self.human.j_head_z, self.np_random.uniform(-30, 30))]
        self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None)

        self.init_robot_pose([], [], [], [], collision_objects=[self.human, self.table, self.object])

        p.resetDebugVisualizerCamera(cameraDistance=2.50, cameraYaw=55, cameraPitch=-30, cameraTargetPosition=[-0.2, 0, 0.75], physicsClientId=self.id)
        p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=10, physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        self.init_env_variables()
