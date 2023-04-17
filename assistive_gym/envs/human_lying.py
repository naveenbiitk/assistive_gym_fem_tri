import os
from gym import spaces
import numpy as np
import pybullet as p
import smplx
import pickle
import torch

from .env import AssistiveEnv
from .agents import human
from .agents.human import Human
from .agents.human_mesh import HumanMesh
from scipy.spatial.transform import Rotation as R

human_controllable_joint_indices = human.motion_right_arm_joints  #human.right_arm_joints 
#human_controllable_joint_indices = human.head_joints
#human_controllable_joint_indices = human.right_arm_joints  #human.head_joints  #head_joints torso_joints


class HumanLyingEnv(AssistiveEnv):
    def __init__(self):
        super(HumanLyingEnv, self).__init__(robot=None, human=Human(human_controllable_joint_indices, controllable=True), task='human_lying', obs_robot_len= (0), obs_human_len=(16))
        self.f_name = '/nethome/nnagarathinam6/hrl_git/assistive_gym_fem_tri/examples/optimal_frame_lying/data/smpl_pkl_1/smpl_smpl_postures22_1.pkl'
        self.sample_pkl = 1

    def step(self, action):
        if self.human.controllable:
            h_action = action*20
            #print('--------action---------')
            #action = np.concatenate([action['robot'], action['human']])
        
        #print('----action-------',action)
        #h_action = action['human']
        self.take_step(h_action)

        obs = self._get_obs()

        end_effector_velocity = np.linalg.norm(self.human.get_velocity(self.human.right_wrist))
        
        wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]

        dist_gap = np.linalg.norm( np.array(wrist_pos)- np.array(self.target_pos) )
        
        done = self.iteration >= 100
        if dist_gap > 0.08:
            reward_abs = -0.5
            reward_time = -self.iteration
        else:
            reward_abs = +500000
            reward_time = 0
            done = True

        
        reward_action = -np.linalg.norm(action)

        reward = -self.config('distance_gap')*dist_gap + self.config('reward_abs')*reward_abs + self.config('time_w')*reward_time + self.config('action_w')*reward_action

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
        super(HumanLyingEnv, self).reset()

        #self.f_name = 'smpl_postures_pkl_25/smpl_postures'+ str(self.sample_pkl) +'_25.pkl'
        #self.f_name = 'smpl_pkl/smpl_postures'+ '6' +'_5.pkl'
        with open(self.f_name, 'rb') as handle:
            data1 = pickle.load(handle)

        self.human.body_shape = torch.Tensor(data1['betas'])

        self.build_assistive_env(furniture_type='bed',human_impairment='none',fixed_human_base=False)
        self.furniture.set_on_ground()
        self.furniture.set_friction(self.furniture.base, friction=5)

        #self.build_assistive_env(furniture_type=None, human_impairment='none')
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        
        #joints_positions = [(self.human.j_right_shoulder_x, 30)]
        joints_positions = []
        self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=0.1, reactive_gain=0.11)
        #self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=1, reactive_gain=0.11)
        self.human.set_base_pos_orient([0.0, 0.0, 1.199], [-np.pi/2.0, 0, 0])

        self.setup_camera_rpy(camera_target=[0, 0, 0.305+2.101], distance=0.01, rpy=[0, -90, 0], fov=60, camera_width=640, camera_height=1080)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        self.henry_joints = self.pose_reindex()
        #human_jts = self.human_pose_smpl_format()

        henry_jts = self.henry_joints
        #self.scipy_optimizer(henry_jts)
        #sol_point = self.cma_optimizer()

        opts_joints = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

        #self.human.set_joint_angles(opts_joints, sol_point)
        #human_jts = self.human_pose_smpl_format()

        p.setGravity(0, 0, -10, physicsClientId=self.id)
        self.human.set_gravity(0, 0, -10)

        #joint_pose = p.calculateInverseKinematics2(self.human.body, tar_toe_pos)
        self.convert_smpl_body_to_gym()
        for _ in range(12):
            p.stepSimulation(physicsClientId=self.id)
            self.convert_smpl_body_to_gym()
        
        # # Lock the person in place
        self.human.control(self.human.all_joint_indices, self.human.get_joint_angles(), 0.025, 5)
        self.human.set_mass(self.human.base, mass=0)
        self.human.set_base_velocity(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])

        for _ in range(80):
            p.stepSimulation(physicsClientId=self.id)
            #self.convert_smpl_body_to_gym()

        print('Completed reset function')
        self.generate_target()
        #self.init_env_variables()
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


    def pose_reindex(self):

        smpl_pose_jt_1 =  self.load_smpl_model()
        self.human_pos_offset, self.human_orient_offset = self.human.get_base_pos_orient()

        joints_1 = smpl_pose_jt_1.dot(R.from_euler('x', -90, degrees=True).as_matrix())
        smpl_pose_jt_2 = joints_1.dot(R.from_quat(self.human_orient_offset).as_matrix())
        smpl_pose_jt_2 = smpl_pose_jt_2.dot(R.from_euler('y', 180, degrees=True).as_matrix())
        smpl_pose_jt = smpl_pose_jt_2.dot(R.from_euler('z', 180, degrees=True).as_matrix())
        
        return self.smpl_agym_map(smpl_pose_jt)



    def load_smpl_model(self):
        
        directory='/nethome/nnagarathinam6/hrl_git/assistive-gym-fem/assistive_gym/envs/assets'
        model_folder = os.path.join(directory, 'smpl_models')
        model = smplx.create(model_folder, model_type='smpl', gender=self.human.gender)

        print('Smple sample', self.sample_pkl)
        with open(self.f_name, 'rb') as handle:
            data1 = pickle.load(handle)

        df = torch.Tensor(data1['body_pose'])
        dt = torch.reshape(df, (1, 23, 3))
        db = dt[:,:21,:]
        
        orient_tensor = torch.Tensor(data1['global_orient'])
        self.orient_body = orient_tensor.numpy()

        body_pose = np.zeros((1,23*3))
        self.smpl_body_pose = dt[0].numpy()

        output = model(betas=torch.Tensor(data1['betas']), body_pose=data1['body_pose'], return_verts=True)
        #output = model(betas=torch.Tensor(data1['betas']), body_pose=torch.Tensor(body_pose), return_verts=True)

        joints = output.joints.detach().cpu().numpy().squeeze()
        #print('output joint', joints.shape, joints)
        return joints
            
##--------------------------------------


    def convert_smpl_body_to_gym(self):

        smpl_pose_jt_1 = self.smpl_body_pose  #self.smpl_agym_map(self.smpl_body_pose)
        
        print('orient_body', self.orient_body)
        #smpl_pose_jt_1 = smpl_pose_jt_1.dot(R.from_euler('x', -90, degrees=True).as_matrix())
        smpl_pose_jt_1 = smpl_pose_jt_1.dot(R.from_quat(self.human_orient_offset).as_matrix())
        #smpl_pose_jt_1 = smpl_pose_jt_1.dot(R.from_euler('y', 180, degrees=True).as_matrix())
        #smpl_pose_jt_1 = smpl_pose_jt_1.dot(R.from_euler('z', 180, degrees=True).as_matrix())
        
        smpl_bp = smpl_pose_jt_1
 
        self.human.set_base_pos_orient([0, 0.0, 0.90], [-np.pi/2, self.orient_body[0,2],0])

        opts_joints = [ self.human.j_head_x, self.human.j_head_y, self.human.j_head_z,self.human.j_neck,
                        self.human.j_waist_x, self.human.j_waist_y, self.human.j_waist_z ]

        cor_angles = [  smpl_bp[14,0],smpl_bp[14,1],smpl_bp[14,2],smpl_bp[11,0],
                        smpl_bp[2,0],smpl_bp[2,1],smpl_bp[2,2] ]

        self.human.set_joint_angles(opts_joints, cor_angles)       

        opts_joints = [ self.human.j_left_hip_x, self.human.j_left_hip_y, self.human.j_left_hip_z, self.human.j_left_knee ,
                        self.human.j_left_ankle_x, self.human.j_left_ankle_y, self.human.j_left_ankle_z ]

        cor_angles = [smpl_bp[0,0],smpl_bp[0,1],smpl_bp[0,2],smpl_bp[3,0],
                      smpl_bp[6,0],smpl_bp[6,1],smpl_bp[6,2]]

        self.human.set_joint_angles(opts_joints, cor_angles)

        opts_joints = [ self.human.j_right_hip_x, self.human.j_right_hip_y, self.human.j_right_hip_z, self.human.j_right_knee, 
                        self.human.j_right_ankle_x, self.human.j_right_ankle_y, self.human.j_right_ankle_z]

        cor_angles = [smpl_bp[1,0],smpl_bp[1,1],smpl_bp[1,2],smpl_bp[4,0],
                      smpl_bp[7,0],smpl_bp[7,1],smpl_bp[7,2]]

        self.human.set_joint_angles(opts_joints, cor_angles)

        opts_joints = [ self.human.j_left_pecs_x, self.human.j_left_pecs_y, self.human.j_left_pecs_z ,
                        self.human.j_left_shoulder_x, self.human.j_left_shoulder_y, self.human.j_left_shoulder_z ,
                        self.human.j_left_elbow, self.human.j_left_forearm ]
        

        cor_angles_left = [  -smpl_bp[12,2]-1.57,smpl_bp[12,1],smpl_bp[12,0],
                             ((-smpl_bp[15,2])-1.57),smpl_bp[15,1],-(smpl_bp[15,0]),
                             smpl_bp[17,2],smpl_bp[19,1]]

 
        self.human.set_joint_angles(opts_joints, cor_angles_left)
 
        opts_joints = [ self.human.j_right_pecs_x, self.human.j_right_pecs_y, self.human.j_right_pecs_z ,
                        self.human.j_right_shoulder_x, self.human.j_right_shoulder_y, self.human.j_right_shoulder_z,
                        self.human.j_right_elbow, self.human.j_right_forearm]
        

        
        ck = smpl_bp[16,0] if smpl_bp[16,0]<0 else smpl_bp[16,0]+1.57

        cor_angles_right = [smpl_bp[13,2],smpl_bp[13,1],smpl_bp[13,0],
                            1.57-smpl_bp[16,2],smpl_bp[16,1],ck,
                            -smpl_bp[18,2],smpl_bp[20,1]]


        self.human.set_joint_angles(opts_joints, cor_angles_right)
        print('right hand cor_angles', np.array(cor_angles_right)*180/3.14)



    def smpl_agym_map(self, smpl_pose_jt):

        agym_jt_smpl = np.zeros((20,3))

        agym_jt_smpl[0,:] = smpl_pose_jt[15,:] + np.array([ 0.00000000e+00 , 3.72529017e-09, -3.14101279e-02]) 
        agym_jt_smpl[1,:] = smpl_pose_jt[12,:] + np.array([0.00000000e+00 ,1.86264505e-09 ,7.45058060e-09]) 
        agym_jt_smpl[2,:] = smpl_pose_jt[3,:]  + np.array([ 0.00000000e+00 , 3.72529028e-09 ,-2.98023224e-08])
        agym_jt_smpl[3,:] = smpl_pose_jt[0,:]  + np.array([0., 0., 0.])
        agym_jt_smpl[4,:] = smpl_pose_jt[6,:]  + np.array([0.00000000e+00 ,3.72529026e-09 ,2.98023224e-08])
        agym_jt_smpl[5,:] = smpl_pose_jt[9,:]  + np.array([ 0.00000000e+00 ,-5.55111512e-17 ,-7.45058060e-09])
        #arms
        agym_jt_smpl[6,:] = smpl_pose_jt[17,:] + np.array([0.00000000e+00 ,2.79396764e-09 ,0.00000000e+00])
        agym_jt_smpl[7,:] = smpl_pose_jt[16,:] + np.array([ 0.00000000e+00 , 2.79396763e-09 ,-2.98023224e-08])
        agym_jt_smpl[8,:] = smpl_pose_jt[19,:] + np.array([ -0.00457314, -0.0248985 ,  0.02630745])
        agym_jt_smpl[9,:] = smpl_pose_jt[18,:] + np.array([-0.00560603, -0.03206245  ,0.03407168])
        agym_jt_smpl[10,:] = smpl_pose_jt[21,:] + np.array([-0.02053231 ,-0.02895589 ,-0.0012251 ]) 
        agym_jt_smpl[11,:] = smpl_pose_jt[20,:] + np.array([ 0.01074898, -0.02654349  ,0.00240007]) 
        agym_jt_smpl[12,:] = smpl_pose_jt[14,:] + np.array([0.00000000e+00 ,3.72529022e-09 ,7.45058060e-08]) 
        agym_jt_smpl[13,:] = smpl_pose_jt[13,:] + np.array([0.00000000e+00, 3.72529022e-09 ,2.98023224e-08]) 
        #legs
        agym_jt_smpl[14,:] = smpl_pose_jt[5,:] + np.array([ 0.00000000e+00 , 3.72529040e-09 ,-6.24269247e-03])
        agym_jt_smpl[15,:] = smpl_pose_jt[4,:] + np.array([ 0.00000000e+00,  1.86264525e-09 ,-4.67756391e-03])
        agym_jt_smpl[16,:] = smpl_pose_jt[2,:] + np.array([0.00000000e+00 ,3.72529033e-09 ,7.57336617e-04])
        agym_jt_smpl[17,:] = smpl_pose_jt[1,:] + np.array([ 0.00000000e+00 , 1.38777878e-17 ,-4.67753410e-03])
        agym_jt_smpl[18,:] = smpl_pose_jt[8,:] + np.array([ 0.00000000e+00 , 3.72529049e-09 ,-1.32426843e-02])
        agym_jt_smpl[19,:] = smpl_pose_jt[7,:] + np.array([ 0.00000000e+00 , 1.94289029e-16 ,-4.67755646e-03])

        agym_jt_smpl = agym_jt_smpl-agym_jt_smpl[3]+np.array(self.human_pos_offset)+np.array([0.0,0.0,+0.3]) #base pose
        
        return  agym_jt_smpl
