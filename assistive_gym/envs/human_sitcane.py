import os, time
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import smplx
import pickle
import torch
from scipy import optimize
import cma

from .env import AssistiveEnv
from .agents.human_mesh import HumanMesh
from .agents import human
from .agents.human import Human
from scipy.spatial.transform import Rotation as R

#from bed_pose_optimization_fun import *

human_controllable_joint_indices = human.motion_right_arm_joints 
class HumanSitcaneEnv(AssistiveEnv):
    def __init__(self, use_mesh=False):
        super(HumanSitcaneEnv, self).__init__(robot=None, human=Human(human_controllable_joint_indices, controllable=True), task='human_sitcane', obs_robot_len=0, obs_human_len=0, frame_skip=5, time_step=0.02)
        self.use_mesh = use_mesh
        self.sample_pkl = 1
        self.sample_pkl_list = [4,5,13,15,19,21,25,39,41,48,49,56,58,59,63,64,70,73,81,91,92,97]
  
    def set_file_name(self, pkl_file_name):
        self.f_name = pkl_file_name

    def step(self, action):
        #action = np.zeros([5])
        
        self.take_step(action, action_multiplier=0.003)

        return np.zeros(1), 0, False, {}


    def _get_obs(self, agent=None):
        return np.zeros(1)


    def reset(self):
        super(HumanSitcaneEnv, self).reset()

        self.build_assistive_env(furniture_type='wheelchair', fixed_human_base=True, gender='male', human_impairment='none')

        # self.show_bounds( arm_length=0.5, bnd_point = [0,0,0])

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-80, cameraTargetPosition=[0.12, 0, 1.5/2.0], physicsClientId=self.id)
        
        # Setup human in the air and let them settle into a Siting pose on the bed
        #joints_positions = [(self.human.j_right_elbow, -20), (self.human.j_left_elbow, -90), (self.human.j_right_hip_x, -90), (self.human.j_right_knee, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee, 80)]
        #joints_positions += [(self.human.j_head_x, self.np_random.uniform(-30, 30)), (self.human.j_head_y, self.np_random.uniform(-30, 30)), (self.human.j_head_z, self.np_random.uniform(-30, 30))]
        
        #joints_positions += [(self.human.j_left_shoulder_x, -70),(self.human.j_left_shoulder_y, -70),(self.human.j_left_shoulder_z, -70), (self.human.j_left_elbow, -120)]
        #self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None)
        #self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=0.1)
        # joints_positions = [(self.human.j_right_elbow, -90), (self.human.j_left_elbow, -90), (self.human.j_waist_y, -10),(self.human.j_head_x, 20),
        # (self.human.j_right_hip_x, -80), (self.human.j_right_knee, 80), (self.human.j_left_hip_x, -80), (self.human.j_left_knee, 80)]
        # self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=None)
        
        human_height, human_base_height = self.human.get_heights()
        self.human.set_base_pos_orient([0, 0.03, human_base_height-0.40], [0, 0, 0, 1])
        
        joints_positions = [(self.human.j_right_elbow, -90), (self.human.j_left_elbow, -90), (self.human.j_right_hip_x, -90), (self.human.j_right_knee, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee, 80)]
        joints_positions += [(self.human.j_waist_x, -10)]
        joints_positions += [(self.human.j_head_x, -self.np_random.uniform(0, 10)), (self.human.j_head_y, -self.np_random.uniform(0, 10)), (self.human.j_head_z, -self.np_random.uniform(0, 10))]
        
        # j_right_pecs_x, j_right_pecs_y, j_right_pecs_z, j_right_shoulder_x, j_right_shoulder_y, j_right_shoulder_z, j_right_elbow, j_right_forearm, j_right_wrist_x, j_right_wrist_y
        # human_start = [-1.43457602e-02,  4.88734760e-04,  5.31665415e-01,  1.34692067e-01, -9.64689935e-02, -2.42673015e-01, -1.64185327e+00,  2.51781710e-02,  6.91561229e-03,  1.17776584e-01]
        right_joint_list = [self.human.j_right_pecs_x, self.human.j_right_pecs_y, self.human.j_right_pecs_z,self.human.j_right_shoulder_x,self.human.j_right_shoulder_y, self.human.j_right_shoulder_z, self.human.j_right_elbow, self.human.j_right_forearm, self.human.j_right_wrist_x, self.human.j_right_wrist_y]
        
        # start
        right_arm_angles = [0.0,  0.0,  5.31665415e-01,  1.34692067e-01, -9.64689935e-02, -2.42673015e-01, -1.64185327e+00,  0.0,  0.0,  0.0]
        
        #end
        # right_arm_angles = [0.0,  0.0, 0.0,  0.6285238,  -1.2184946,   0.0,  -2.2070327,  0.0, 0.0, 0.0]

        right_arm_pos = [(right_joint_list[i], right_arm_angles[i]*180/np.pi) for i in range(len(right_joint_list))]
        joints_positions += right_arm_pos

        self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=1, reactive_gain=0.11)
        #self.human.set_base_pos_orient([0, 0.2, 0.80], [-np.pi/2, -np.pi/2, np.pi/2])
        
        # Add small variation in human joint positions
        #motor_indices, motor_positions, motor_velocities, motor_torques = self.human.get_motor_joint_states()
        #self.human.set_joint_angles(motor_indices, motor_positions+self.np_random.uniform(-0.1, 0.1, size=len(motor_indices)))

        for joints_j in self.human.controllable_joint_indices:
            self.human.enable_force_torque_sensor(joints_j) 

        #self.siting_posture()
        self.setup_camera_rpy(camera_target=[0, 0, 0.305+2.101], distance=0.01, rpy=[0, -90, 0], fov=60, camera_width=640, camera_height=1080)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        #p.setGravity(0, 0, -10, physicsClientId=self.id)
        #self.human.set_gravity(0, 0, 0)
 
        for _ in range(42):
            p.stepSimulation(physicsClientId=self.id)
        
        # Initialize the tool in the robot's gripper
        self.tool.init(self.human, self.task, self.directory, self.id, self.np_random, right=False, mesh_scale=[0.001]*3)

        # # Lock the person in place
        self.human.control(self.human.all_joint_indices, self.human.get_joint_angles(), 0.025, 5)
        #self.human.set_mass(self.human.base, mass=0)
        self.human.set_base_velocity(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])

        for _ in range(20):
            p.stepSimulation(physicsClientId=self.id)
            #self.convert_smpl_body_to_gym()

        #print('Completed reset function')

        #self.init_env_variables()
        return self._get_obs()


    def siting_posture(self):
        joints_positions = [(self.human.j_right_elbow, -40), (self.human.j_left_elbow, -30), (self.human.j_right_hip_x, -90), (self.human.j_right_knee, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee, 80)]
        joints_positions += [(self.human.j_head_x, self.np_random.uniform(-30, 30)), (self.human.j_head_y, self.np_random.uniform(-30, 30)), (self.human.j_head_z, self.np_random.uniform(-30, 30))]
        joints_positions += [(self.human.j_waist_x, 0),(self.human.j_waist_y, -10), (self.human.j_waist_z, 0)]
        #self.human.j_waist_x, self.human.j_waist_y, self.human.j_waist_z 

        self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=None)





'''

Change the cane model (not working)

How to optimize simulation and plot the results for cane

Cane is attached to the hand

Plot the 2d map how is the different from the small object handover

Plot from top view, how the small object and the cane is different

how to do that collision detection (reachability)

set of x,y,z points

for loop of the cane see
    1. set the base-orient  
    env.tool.set_base_pos_orient(transform_pos, transform_orient)
    
    2. collision detection
    see whether cane is colliding with the chair and human

    3. ik/ score for reaching the point
    the function is there object handover 

    4. Score that in list
    save the score

    5. What to plot one of the good candidate is low score

    6. What about frames/ How to make connections?
    frames/how to make connections

'''





