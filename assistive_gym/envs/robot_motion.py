import numpy as np
import pybullet as p
import optparse
from scipy.spatial.transform import Rotation as R
import os
import cv2

from assistive_gym.envs.agents.furniture import Furniture

from .env import AssistiveEnv
from .agents import furniture
from .agents.furniture import Furniture
from .util import reward_base_direction
from .util import reward_tool_direction
#from .util import generate_line
#from .util import generate_line_hand
from scipy.spatial.transform import Rotation
# from rlhf import RLHF_Model
from assistive_gym.envs.robot_rlhf.rlhf import RLHF_Model
# from loader import RLHF_Dataset
from assistive_gym.envs.robot_rlhf.loader import RLHF_Dataset
from assistive_gym.envs.robot_rlhf.run_model import *
from assistive_gym.envs.robot_rlhf.rlhf_utils import *
import torch

class RobotMotionEnv(AssistiveEnv):

    def __init__(self, robot, human):
        super(RobotMotionEnv, self).__init__(robot=robot, human=human, task='robot_motion', obs_robot_len=(20), obs_human_len=(19))
        self.phase_of_robot=1
        self.phase_of_human=1
        self.robot_grab_cup_epsilon=0.05
        self.cup_target_epsilon=0.05
        self.human_mouth_epsilon=0.2
        self.human_bowl_epsilon=0.1
        self.cup_top_center_offset = np.array([0, 0, -0.055])
        self.cup_bottom_center_offset = np.array([0, 0, 0.07])
        self.total_reward = 0
        self.max_steps  = 120

        self.rlhf = False # default False
        if self.rlhf:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.rlhf_model = RLHF_Model(input_size=23*60, output_size=1).to(self.device)
            self.rlhf_model.load_state_dict(torch.load('/nethome/nnagarathinam6/hrl_git/assistive_gym_fem_tri/assistive_gym/envs/robot_rlhf/checkpoints/model_4_1k_best.pt', map_location=torch.device('cpu')))
                

    def robot_reward(self):

        robot_wrist_pos,_orient_ = self.robot.get_pos_orient(self.robot.left_end_effector)
        robo_wrist_pos_np = np.array(robot_wrist_pos)

        cup_pos, cup_orient = self.tool.get_base_pos_orient()
        cup_pos, cup_orient = p.multiplyTransforms(cup_pos, cup_orient, [0, 0.06, 0], self.get_quaternion([np.pi/2.0, 0, 0]), physicsClientId=self.id)
        cup_top_center_pos, _ = p.multiplyTransforms(cup_pos, cup_orient, self.cup_top_center_offset, [0, 0, 0, 1], physicsClientId=self.id)

        cup_wrist_dist = np.linalg.norm(np.array(cup_top_center_pos)-robo_wrist_pos_np)
        cup_goal_dist = np.linalg.norm(self.target_2_pose-cup_top_center_pos)
        reward_robot = -self.config('distance_weight')*(cup_wrist_dist + cup_goal_dist)

        cup_angles = Rotation.from_quat(cup_orient)
        cup_angles = cup_angles.as_euler('xyz', degrees=True)

        if cup_goal_dist < 0.125:
            reward_robot += 75
            reward_robot += cup_wrist_dist*50

        if self.rlhf:
            reward_robot = 0



        return reward_robot         


    def reward_function(self, prev_state, state):
        # Constants
        ALPHA = 1  # weight for distance to object
        BETA = 1   # weight for distance to target
        GAMMA = 0.5  # weight for the gripper status
        PENALTY = -0.01  # penalty for taking unnecessary steps
        success_threshold = 0.075  # threshold for success
        success_reward = 1  # reward for success
        let_go_reward = 0.25  # reward for letting go of cup at target
        failure_penalty = -100  # penalty for failure
        # Extract state variables
        gripper_position, gripper_status, object_top_position, object_bottom_position, target_position = state

        # # Extract next state variables
        # next_gripper_position, next_gripper_status, next_object_position, next_target_position = next_state

        # # Calculate distances
        # distance_to_object = np.linalg.norm(gripper_position - object_position)
        # distance_to_target = np.linalg.norm(object_position - target_position)
        # next_distance_to_object = np.linalg.norm(next_gripper_position - next_object_position)
        # next_distance_to_target = np.linalg.norm(next_object_position - next_target_position)

        reward = 0
        on_target = np.linalg.norm(object_bottom_position - target_position) < success_threshold

        # minimize distance between gripper and cup
        reward += -ALPHA * np.linalg.norm(gripper_position - object_top_position)
        
        if on_target:
            # constant reward for reaching target
            reward += success_reward
            # constant reward for letting go of cup at target
            reward += 2 * ALPHA * np.linalg.norm(gripper_position - object_top_position)
        else:
            # constant reward for grabbing cup
            reward += GAMMA * gripper_status 

        # minimize distance between cup and target
        reward += -BETA * np.linalg.norm(object_bottom_position - target_position)

        print('object_target_dist: ', np.linalg.norm(object_bottom_position - target_position) )
        

        

        # constant penalty for taking unnecessary steps
        reward += PENALTY

        # # constant penalty for failure
        # if self.iteration >= self.max_steps-1:
        #     reward += failure_penalty

        # # Calculate rewards based on distances and gripper status
        # reward = -ALPHA * (next_distance_to_object - distance_to_object) \
        #             -BETA * (next_distance_to_target - distance_to_target) \
        #             -GAMMA * (next_gripper_status - gripper_status) \
        #             + PENALTY

        

        # # Check for terminal conditions
        # is_success = next_distance_to_target < success_threshold and next_gripper_status == 1
        # is_failure = self.iteration >= self.max_steps-1

        # if is_success:
        #     reward += success_reward
        # elif is_failure:
        #     reward += failure_penalty

        print('current reward', reward)
        return reward

            
    def obs_state(self):
        gripper_position,_ = self.robot.get_pos_orient(self.robot.left_end_effector)
        gripper_position_np = np.array(gripper_position)

        cup_pos, cup_orient = self.tool.get_base_pos_orient()
        cup_pos, cup_orient = p.multiplyTransforms(cup_pos, cup_orient, [0, 0.06, 0], self.get_quaternion([np.pi/2.0, 0, 0]), physicsClientId=self.id)
        cup_top_center_pos, _ = p.multiplyTransforms(cup_pos, cup_orient, self.cup_top_center_offset, [0, 0, 0, 1], physicsClientId=self.id)
        cup_bottom_center_pos, _ = p.multiplyTransforms(cup_pos, cup_orient, self.cup_bottom_center_offset, [0, 0, 0, 1], physicsClientId=self.id)

        object_top_position = np.array(cup_top_center_pos)
        object_bottom_position = np.array(cup_bottom_center_pos)
        target_position = np.array(self.target_2_pose)

        # print('-----target--position-----', target_position)
        
        # self.create_sphere(radius=0.02, mass=0.0, pos=object_position, visual=True, collision=False, rgba=[1, 0, 1, 1])# cyan bowl
        # self.create_sphere(radius=0.02, mass=0.0, pos=target_position, visual=True, collision=False, rgba=[1, 0, 1, 1])# cyan bowl
        # self.create_sphere(radius=0.02, mass=0.0, pos=gripper_position, visual=True, collision=False, rgba=[1, 0, 1, 1])# cyan bowl


        gripper_status = 1 if np.linalg.norm(gripper_position_np - object_top_position) < self.robot_grab_cup_epsilon else 0

        return [gripper_position, gripper_status, object_top_position, object_bottom_position, target_position]


    def step(self, action):

        prev_state = self.obs_state()
        self.take_step(action*1)
        obs = self._get_obs()
        
        state = self.obs_state()
        reward = self.reward_function(prev_state, state)
        self.iteration += 1
        
        done = self.iteration >= self.max_steps

        if self.rlhf:
            reward = 0
            self.actions.append(action)
            self.observations.append(obs)

            if done:
                trajectory = np.concatenate((self.observations, self.actions), axis=1)
                # trajectory = np.zeros((120, 25))
                trajectory = torch.from_numpy(trajectory).unsqueeze(0)
                # trajectory = torch.from_numpy(observations).unsqueeze(0)
                reward = run_model(self.rlhf_model, trajectory, self.device) * 60
                
                # self.rewards = [0] * 120
                # reward = 100
                
            self.rewards.append(reward)
            self.total_reward = sum(self.rewards)

        else:
            self.total_reward = self.total_reward+reward

        if self.gui and self.iteration < 0:
            #print('Task success:', self.task_success, 'Tool force at target:', self.tool_force_at_target, reward_force_scratch)
            print('Iteration:',self.iteration,'Task reward:', self.total_reward)
            print('Current Reward: ', reward)

        #info = {'total_force_on_human': self.total_force_on_human, 'task_success': int(self.task_success >= self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        info = {'task_success': self.task_success, 'action_robot_len': self.action_robot_len,  'obs_robot_len': self.obs_robot_len}

        return obs, reward, done, info

    def _get_obs(self, agent=None):

        robot_wrist_pos,_orient_ = self.robot.get_pos_orient(self.robot.left_end_effector)
        robo_wrist_pos_np = np.array(robot_wrist_pos)
        tool_pos, tool_orient = self.tool.get_base_pos_orient()
        tool_pos_real, tool_orient_real = self.robot.convert_to_realworld(tool_pos, tool_orient)
        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)
        # Fix joint angles to be in [-pi, pi]
        robot_joint_angles = (np.array(robot_joint_angles) + np.pi) % (2*np.pi) - np.pi
        if self.robot.mobile:
            # Don't include joint angles for the wheels
            robot_joint_angles = robot_joint_angles[len(self.robot.wheel_joint_indices):]
        
        robot_wrist_pos,_orient_ = self.robot.get_pos_orient(self.robot.left_end_effector)
        robot_base_pos,robot_orientation = self.robot.get_pos_orient(self.robot.base)

        robot_base_angle = self.robot.get_euler(robot_orientation)
        #print('---Robot base angle___',robot_base_angle*180/np.pi)

        target_pos_real, _ = self.robot.convert_to_realworld(self.target_pos)
        target_pos_real_2, _ = self.robot.convert_to_realworld(self.target_2_pose)
        target_pos_real_3, _ = self.robot.convert_to_realworld(self.target_3_pose)


        #print(np.array([self.iteration, self.phase_of_robot, self.total_force_on_human]),robot_wrist_pos,robot_base_angle, tool_pos_real, tool_pos_real - self.target_2_pose, target_pos_real_2,target_pos_real_3, robot_joint_angles, wrist_pos_real)
        robot_obs = np.concatenate([np.array([self.iteration, self.phase_of_robot]), robo_wrist_pos_np, self.target_2_pose-robo_wrist_pos_np , tool_pos_real, tool_pos_real - self.target_2_pose, target_pos_real_2, robot_joint_angles]).ravel()
        #print('------------------------------------------Robot obs space', len(robot_obs))
         
        return robot_obs




    def reset(self):
        super(RobotMotionEnv, self).reset()

        self.build_assistive_env()

        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        # Update robot and human motor gains
        self.robot.motor_gains = 0.005
        #self.robot.reset_joints()
        self.table = Furniture()
        #self.chair.init(7, env.id, env.np_random, indices=-1)
        self.table.init(furniture_type='table', directory=self.directory, id=self.id, np_random=self.np_random)
        self.table.set_base_pos_orient([0.3, -0.85, 0.0], [0, 0, 0])
        self.table.set_gravity(0, 0, -9.8)
        self.robot.set_gravity(0, 0, -9.8)
        #self.robot.set_whole_body_frictions(lateral_friction=100, spinning_friction=100, rolling_friction=100)

        self.generate_target()

        p.resetDebugVisualizerCamera(cameraDistance=1.10, cameraYaw=40, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.75], physicsClientId=self.id)
        #p.resetDebugVisualizerCamera(cameraDistance=1.10, cameraYaw=55, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.75], physicsClientId=self.id)
        #self.setup_camera_rpy(camera_target=[-0.4, -0.1, 0.75], distance=1.8, rpy=[0, -50, -25], fov=60, camera_width=1280, camera_height=1620)
        #self.setup_camera_rpy(camera_target=[-0.2, 0.0, 0.75], distance=1.8, rpy=[0, -40, 55], fov=60, camera_width=1280, camera_height=1620)  #camera_width=1920//2, camera_height=1080//2
        # p.resetDebugVisualizerCamera(cameraDistance=1.75, cameraYaw=-25, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.4], physicsClientId=self.id)
        #self.setup_camera_rpy(camera_target=[-0.9, -0.4, 0.75], distance=1.5, rpy=[0, -45, 65], fov=60, camera_width=1280, camera_height=1280)  #camera_width=1920//2, camera_height=1080//2
        ct = [-0.4, 0, 0.95]
        pan_ang = [0, -45, 30]
        self.setup_camera_rpy(camera_target=ct, distance=1.7, rpy=pan_ang, fov=60, camera_width=1280//4, camera_height=1620//4)  #camera_width=1920//2, camera_height=1080//2
        print('---camera target', ct,'--ang--',pan_ang )
        self.tool.init(self.robot, self.task, self.directory, self.id, self.np_random, right=True, mesh_scale=[0.045]*3, alpha=0.75)
        self.robot.skip_pose_optimization = True
        target_ee_pos = np.array([-0.1, 0.1, 0.5]) 
        target_ee_orient = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task])
        #self.init_robot_pose(target_ee_pos, target_ee_orient, [(target_ee_pos, target_ee_orient), (self.target_pos, None)], [(self.target_pos, target_ee_orient)], arm='right', tools=[self.tool], collision_objects=[self.human, self.furniture])

        pos = [-0.625, -0.64, 0.1]
        # pos = [-1.825, -0.5, 0.1]
        #orient = [0, 0, (np.pi/2.0)+np.random.uniform() ]
        orient = [0, 0, np.pi/2.0 ]
        self.robot.set_base_pos_orient(pos, orient)
        self.robot.randomize_init_joint_angles(self.task)
        self.robot
        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.right_gripper_indices, self.robot.gripper_pos[self.task], set_instantly=True)

        # self.bowl = Furniture()
        # self.bowl.init('bowl', self.directory, self.id, self.np_random)

        # if not self.robot.mobile:
        #     self.robot.set_gravity(0, 0, 0)
       
        self.tool.set_gravity(0, 0, -9)


        # Enable rendering
        #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # Drop food in the spoon
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.id)

        #print('Robot is mobile or not', self.robot.mobile)
        p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=10, physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        self.init_env_variables()
        print('---sample--',self.observation_space.sample())

        if self.rlhf:
            self.actions = []
            self.observations = []
            self.rewards = []

        return self._get_obs()



    def generate_target(self):
        # Generate target
        #self.target = Furniture()
        #self.target.init('target', self.directory, self.id, self.np_random)
        self.target_pos = np.array([0.007, -0.56, 0.73])
        #self.target = self.create_sphere(radius=0.03, mass=0.001, pos=self.target_pos, visual=True, collision=True, return_collision_visual=False)
        #self.target.set_whole_body_frictions(lateral_friction=100, spinning_friction=100, rolling_friction=100)
        #self.target.set_pos(self.target_pos, self.target.base)
        #self.target.set_gravity(0, 0, 0)
        #self.target.set_color([0, 1, 0, 1])
        target_pos_2 = [0.007, -0.56, 0.73]
        self.target_2_pose = np.array(target_pos_2) + self.np_random.uniform(-0.05, 0.05, size=3)
        self.target_3_pose = self.target_2_pose
        self.create_sphere(radius=0.02, mass=0.0, pos=target_pos_2, visual=True, collision=False, rgba=[0, 1, 1, 1])# cyan bowl

    
    def get_depth_image(self):
        # * take image after reward computed
        img, depth = self.get_camera_image_depth()

        far = 2.406
        near = 1.406

        a_ = (far-near)/(np.max(depth) - np.min(depth))
        b_ = - (far*np.min(depth)-near*np.max(depth))/(np.max(depth) - np.min(depth))
        depth = depth*(a_)+b_

        depth = depth*1000

        #depth = depth[50:178, 73:127]

        #filename='/nethome/nnagarathinam6/Documents/joint_reaching_evaluation/'
        #outfile = filename + "after_depth" + str(1) + ".npy"
        #np.save(outfile, depth)
        
        #file = filename + "depth" + str(1) + ".png"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('color', img)
        cv2.waitKey(0) 

        return depth




