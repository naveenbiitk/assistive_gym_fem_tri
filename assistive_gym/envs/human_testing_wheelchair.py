import os
from gym import spaces
import numpy as np
import pybullet as p

from .env import AssistiveEnv
from .agents import human
from .agents.human import Human
from .agents.human_mesh import HumanMesh


#human_controllable_joint_indices = human.motion_right_arm_joints  #human.right_arm_joints 
#human_controllable_joint_indices = human.head_joints
human_controllable_joint_indices = human.right_arm_joints  #human.head_joints  #head_joints torso_joints


class HumanTestingEnv(AssistiveEnv):
    def __init__(self):
        super(HumanTestingEnv, self).__init__(robot=None, human=Human(human_controllable_joint_indices, controllable=True), task='human_testing', obs_robot_len= (0), obs_human_len=(16))



    def step(self, action):
        if self.human.controllable:
            h_action = action
            #print('--------action---------')
            #action = np.concatenate([action['robot'], action['human']])
        
        #print('----action-------',action)
        #h_action = action['human']
        self.take_step(h_action)

        obs = []
        # obs = self._get_obs()

        # end_effector_velocity = np.linalg.norm(self.human.get_velocity(self.human.right_wrist))
        
        # wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]

        # dist_gap = np.linalg.norm( np.array(wrist_pos)- np.array(self.target_pos) )
        
        done = self.iteration >= 100
        # if dist_gap > 0.08:
        #     reward_abs = -0.5
        #     reward_time = -self.iteration
        # else:
        #     reward_abs = +500000
        #     reward_time = 0
        #     done = True


        
        #reward_action = -np.linalg.norm(action)

        #reward = -self.config('distance_gap')*dist_gap + self.config('reward_abs')*reward_abs + self.config('time_w')*reward_time + self.config('action_w')*reward_action

        reward = 1
        self.total_reward = self.total_reward+reward
        
        # if self.gui and self.iteration > 0:
            # print('Task success:', self.task_success, 'Tool force at target:', self.tool_force_at_target, reward_force_scratch)
            # print('Iteration:',self.iteration,'Task reward:', self.total_reward)
            # print('Current Reward: ', reward)

        
        info = {'total_force_on_human': self.total_reward}        #{'robot': reward, 'human': reward}
        return obs, reward,  done, info


    def _get_obs(self, agent=None):

        human_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
        
        wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]
        wrist_pos_human, _ = self.human.convert_to_realworld(wrist_pos)

        elbow_pos, elbow_orient_quat = self.human.get_pos_orient(self.human.right_elbow)
        elbow_pos_human, _ = self.human.convert_to_realworld(elbow_pos)
 
        human_obs = np.concatenate([np.array([self.iteration]),self.target_pos, human_joint_angles, wrist_pos_human, elbow_pos_human]).ravel()

        #print(human_obs)
        human_obs = human_obs.astype('float32')
        return human_obs
        robot_obs = []
        if self.human.controllable:
            return {'robot': robot_obs, 'human': human_obs}

        return {'robot': robot_obs, 'human': human_obs}



    def reset(self):
        super(HumanTestingEnv, self).reset()

        self.build_assistive_env(furniture_type='wheelchair',human_impairment='none',fixed_human_base=False)
        self.furniture.set_on_ground()
        self.furniture.set_friction(self.furniture.base, friction=5)

        #self.build_assistive_env(furniture_type=None, human_impairment='none')
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        
        human_height, human_base_height = self.human.get_heights()
        self.human.set_base_pos_orient([0, 0.03, human_base_height-0.35], [0, 0, 0, 1])
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
       
        #self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None)

        self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=1, reactive_gain=0.11)

        p.setGravity(0, 0, -1, physicsClientId=self.id)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=40, cameraPitch=-50, cameraTargetPosition=[1.02, 0, 1.5/2.0], physicsClientId=self.id)
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)

        # Lock human joints and set velocities to 0
        self.generate_target()
        #joints_positions = []
        #self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=None, reactive_gain=0.01)
        #self.human.set_mass(self.human.base, mass=0)
        self.human.set_mass(self.human.base, mass=0)
        self.human.set_base_velocity(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])
        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        # Drop the blanket on the person
        for _ in range(30):
            p.stepSimulation(physicsClientId=self.id)

        for joints_j in self.human.controllable_joint_indices:
            self.human.enable_force_torque_sensor(joints_j) 
        #self.init_env_variables()
        #print('---sample--',self.observation_space.sample())
        #print('--------------')
        return self._get_obs()


    def generate_target(self):

        self.total_reward=0
        
        if self.human.gender == 'male':
            self.limb, length, radius = self.human.right_elbow, 0.157, 0.033
        else:
            self.limb, length, radius = self.human.right_elbow, 0.134, 0.027
           
        self.target_on_arm = self.util.point_on_capsule(p1=np.array([0, 0, -2*length]), p2=np.array([0, 0, -length*2.5]), radius=radius, theta_range=(0, np.pi*2))
        arm_pos, arm_orient = self.human.get_pos_orient(self.limb)
        target_pos_1, target_orient = p.multiplyTransforms(arm_pos, arm_orient, self.target_on_arm, [0, 0, 0, 1], physicsClientId=self.id)

        target_pos_2 = np.array([+0.015, -0.25, 1.23])
        #array([[0.28766633, 0.96236509, 0.32128926]])
        #rnd_array = np.random.rand(3)/5  easy to train
        rnd_array = np.random.rand(3)/3  # difficult to train
        target_pos_2 = target_pos_2-rnd_array
        
        #self.target = [0,0,0]
        self.target = self.create_sphere(radius=0.02, mass=0.0, pos=[0,0,-2], visual=True, collision=False, rgba=[1, 1, 0, 1]) #human hand
        #print('Target pose 1', target_pos_1)
        
        self.target_pos = np.array(target_pos_2) 
        #print('---Target_pose-----', self.target_pos)
        #self.create_sphere(radius=0.02, mass=0.0, pos=target_pos_2, visual=True, collision=False, rgba=[1, 1, 0, 1])# pink robot,cup
        
        self.update_targets()



    def update_targets(self):
        arm_pos, arm_orient = self.human.get_pos_orient(self.limb)
        target_pos_1, target_orient = p.multiplyTransforms(arm_pos, arm_orient, self.target_on_arm, [0, 0, 0, 1], physicsClientId=self.id)
        self.target_pos_1 = np.array(target_pos_1)
        self.target_orient = np.array(target_orient)
        self.target.set_base_pos_orient([0,0,-2], [0, 0, 0, 1])
        #print('Handover pose in main environment ',self.target_pos)
        # target should be wrt hip