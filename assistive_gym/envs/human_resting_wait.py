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
from .naveenb_utils.human_coordinate import *
from pytorch3d import transforms
from pathlib import Path


# pip install pytorch3d cma numpngw

human_controllable_joint_indices = human.motion_right_arm_joints 
class HumanRestingEnv(AssistiveEnv):
    def __init__(self, use_mesh=False):
        super(HumanRestingEnv, self).__init__(robot=None, human=Human(human_controllable_joint_indices, controllable=True), task='human_resting', obs_robot_len=0, obs_human_len=0, frame_skip=5, time_step=0.02)
        self.use_mesh = use_mesh
        self.sample_pkl = 1
        self.sample_pkl_list = [4,5,13,15,19,21,25,39,41,48,49,56,58,59,63,64,70,73,81,91,92,97]
        cur_path = os.path.dirname(__file__)
        self.f_name  = os.path.normpath(os.path.join(cur_path, '../../examples/optimal_frame_lying/data/smpl_bp_ros_smpl_3.pkl'))      
        self.save_fname = "human_joints_simulation_realtime_1.pkl" 
        self.save_fname_png = "human_joints_simulation_realtime_1.png" 
        self.count = 0

    def set_file_name(self, pkl_file_name):
        self.f_name = pkl_file_name

    def step(self, action):
        #action = np.zeros([5])
        self.take_step(action, action_multiplier=0.003)
        self.count += 1
        #print('-----------------------step--------------------------', self.count)
        #self.convert_smpl_body_to_gym()
        if self.count ==3000:
            #save the human pose
            self.save_human_model()

        return np.zeros(1), 0, False, {}

    def save_human_model(self):
        #save the 3D human pose
        joints_3d_h = self.get_human_joint_position()
        dict_item = {'human_joint_3d': joints_3d_h}
        f = open(self.save_fname,"wb")
        print('-----------------------Saved------Human3djoints--------------------')
        pickle.dump(dict_item,f)
        img, depth = self.get_camera_image_depth()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #save the image
        cv2.imwrite(self.save_fname_png, img)


    def _get_obs(self, agent=None):
        #self.human.set_joint_angles([self.human.j_left_elbow], [0.0])
        self.convert_smpl_body_to_gym()
        return np.zeros(1)


    def change_human_pose(self):
        self.load_smpl_model()
        for _ in range(12):
            p.stepSimulation(physicsClientId=self.id)
            self.convert_smpl_body_to_gym()

        for _ in range(32):
            p.stepSimulation(physicsClientId=self.id)

        for _ in range(100):
            st_flag = self.human_bed_collision()
            p.stepSimulation(physicsClientId=self.id)
            if st_flag:
                self.human.control(self.human.all_joint_indices, self.human.get_joint_angles(), 1.125, 1.1)
                #self.human.set_mass(self.human.base, mass=1.1)
                #self.human.set_base_velocity(linear_velocity=[0, 0, 0],angular_velocity=[0, 0, 0])
                break
            #self.human.set_base_velocity(linear_velocity=[0, 0, -9.55],angular_velocity=[0, 0, 0])
            #self.convert_smpl_body_to_gym()
        #self.human.set_joint_angles(self.human.j_left_elbow, 0.0)
        #self.convert_smpl_body_to_gym()
        #print('Completed reset function')
        # self.human.set_mass(self.human.base, mass=100.1)
        # self.human.set_mass(self.human.head, mass=100.1)



    def reset(self):
        super(HumanRestingEnv, self).reset()
        
        #self.f_name = 'smpl_postures_pkl_25/smpl_postures'+ str(self.sample_pkl) +'_25.pkl'
        #self.f_name = 'smpl_pkl/smpl_postures'+ '6' +'_5.pkl'
        with open(self.f_name, 'rb') as handle:
            data1 = pickle.load(handle)

        print('--data1', data1)
        self.human.body_shape = torch.from_numpy(np.array(data1['betas']))
        
        self.build_assistive_env(furniture_type='hospital_bed', fixed_human_base=True, gender='male', human_impairment='none')  #fixedhumanbas-True works well not realistic

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-80, cameraTargetPosition=[0.12, 0, 1.5], physicsClientId=self.id)
        
        self.furniture.set_friction(self.furniture.base, friction=0.5)
        self.human.set_whole_body_frictions(lateral_friction=1, spinning_friction=1, rolling_friction=1)
        self.human.set_gravity(0, 0, -0.1)
        # Setup human in the air and let them settle into a resting pose on the bed
        joints_positions = []
        self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=4.8)
        #self.human.set_base_pos_orient([0, 0.25, 1.12995], [-np.pi/2.0, 0, 0])
        self.human.set_base_pos_orient([0, 0.025, 0.992995], [-np.pi/2.0, 0, 0])
        # Add small variation in human joint positions
        motor_indices, motor_positions, motor_velocities, motor_torques = self.human.get_motor_joint_states()
        self.human.set_joint_angles(motor_indices, motor_positions+self.np_random.uniform(-0.1, 0.1, size=len(motor_indices)))

        for joints_j in self.human.right_arm_joints:
            self.human.enable_force_torque_sensor(joints_j) 

        if self.use_mesh:
            # Replace the capsulized human with a human mesh
            self.human = HumanMesh()
            joints_positions = [(self.human.j_right_shoulder_z, 60), (self.human.j_right_elbow_y, 90), (self.human.j_left_shoulder_z, 60), (self.human.j_left_elbow_y, 90), (self.human.j_right_hip_x, -90), (self.human.j_right_knee_x, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee_x, 80)]
            body_shape = np.zeros((1, 10))
            gender = 'female' # 'random'
            self.human.init(self.directory, self.id, self.np_random, gender=gender, height=None, body_shape=body_shape, joint_angles=joints_positions)
            chair_seat_position = np.array([0, 0.1, 0.55])
            self.human.set_base_pos_orient(chair_seat_position - self.human.get_vertex_positions(self.human.bottom_index), [0, 0, 0, 1])


        self.setup_camera_rpy(camera_target=[0, -0.2, 3.101], distance=0.01, rpy=[0, -90, 0], fov=60, camera_width=1920, camera_height=1080)  #camera_width=1920//2, camera_height=1080//2

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
        self.load_smpl_model()
        #self.henry_joints = self.pose_reindex()
        #human_jts = self.human_pose_smpl_format()

        #henry_jts = self.henry_joints
        #self.scipy_optimizer(henry_jts)
        opts_joints=[self.human.j_left_shoulder_x, self.human.j_left_shoulder_y, self.human.j_left_shoulder_z,self.human.j_left_elbow ]
        #sol_point = self.cma_optimizer(opts_joints=[self.human.j_left_shoulder_x, self.human.j_left_shoulder_y, self.human.j_left_shoulder_z,self.human.j_left_elbow ])
        # opts_joints = [10, 11, 12, 13, 14, 15, 16]
        #self.human.set_joint_angles(opts_joints, sol_point)
        #human_jts = self.human_pose_smpl_format()
        radius = 0.01
        mass = 0.001
        #spheres1 = self.create_spheres(radius=radius, mass=mass, batch_positions=henry_jts, visual=True, collision=False, rgba=[0, 1, 1, 1])
        #spheres2 = self.create_spheres(radius=radius, mass=mass, batch_positions=human_jts, visual=True, collision=False, rgba=[1, 0, 1, 1])

        p.setGravity(0, 0, -1.15, physicsClientId=self.id)
        self.human.set_gravity(0, 0, -1.15)
 
        #joint_pose = p.calculateInverseKinematics2(self.human.body, tar_toe_pos)

        
        # for _ in range(20):
        #     p.stepSimulation(physicsClientId=self.id)
        
        # time.sleep(10)
        # # Lock the person in place
        self.human.control(self.human.all_joint_indices, self.human.get_joint_angles(), 0.025, 5)
        # self.human.set_mass(self.human.base, mass=10.1)
        # self.human.set_mass(self.human.waist, mass=10.1)
        # self.human.set_mass(self.human.head, mass=10.1)
        # self.human.set_mass(self.human.left_knee, mass=10.1)
        # self.human.set_mass(self.human.right_knee, mass=10.1)
        # self.human.set_mass(self.human.left_ankle, mass=10.1)
        # self.human.set_mass(self.human.right_ankle, mass=10.1)
        # self.human.set_mass(self.human.left_shoulder, mass=10.1)
        # self.human.set_mass(self.human.right_shoulder, mass=10.1)
        # self.human.set_mass(self.human.left_wrist, mass=10.1)
        # self.human.set_base_velocity(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])
        #self.human.set_base_velocity(linear_velocity=[0, 0, -0.3], angular_velocity=[0, 0, 0])
        self.convert_smpl_body_to_gym()
        for _ in range(12):
            p.stepSimulation(physicsClientId=self.id)
            self.convert_smpl_body_to_gym()

        for _ in range(32):
            p.stepSimulation(physicsClientId=self.id)

        for _ in range(100):
            st_flag = self.human_bed_collision()
            p.stepSimulation(physicsClientId=self.id)
            #self.human.set_joint_angles(opts_joints, sol_point)
            if st_flag:
                self.human.control(self.human.all_joint_indices, self.human.get_joint_angles(), 0.0125, 5.1)  #reactive_force ,reactive_gain
                #self.human.set_mass(self.human.base, mass=1.1)
                self.human.set_base_velocity(linear_velocity=[0, 0, 0],angular_velocity=[0, 0, 0])
                break
            #self.human.set_base_velocity(linear_velocity=[0, 0, -9.55],angular_velocity=[0, 0, 0])
            #self.convert_smpl_body_to_gym()
    
        #self.convert_smpl_body_to_gym()
        #print('Completed reset function')
        #self.human.set_joint_angles(opts_joints, sol_point)
        self.human.set_mass(self.human.base, mass=10.1)
        self.human.set_mass(self.human.head, mass=10.1)
        # self.sample_pkl = self.sample_pkl+1
        # #self.convert_smpl_body_to_gym()
        # if self.sample_pkl in self.sample_pkl_list:
        #     self.sample_pkl = self.sample_pkl+1

        # mid_angle = np.array([-1.28484584e-14,  1.74532925e-01,  1.24455499e-14, -8.98027383e-02, -1.10828408e+00,  
        #       6.82478992e-14, 0, 0,  7.85398163e-02,  1.74532925e-01])

        # self.human.set_joint_angles(self.human.left_arm_joints, mid_angle)

        #self.init_env_variables()
        return self._get_obs()


    def human_bed_collision(self):
        # when human fall on bed flag will change from false to true
        collision_objects = [self.furniture]
        dists_list = []
        for obj in collision_objects:
            dists = self.human.get_closest_points(obj, distance=0)[-1]
            dists_list.append(dists)
            #print('obj ', obj, ' return ',dists )
        
        collision_flag = True
        if all(not d for d in dists_list):
            collision_flag = False

        #print('--collision: ',collision_flag)    
        return collision_flag


    def load_smpl_model(self):
         
        #print('Smple sample', self.sample_pkl)
        with open(self.f_name, 'rb') as handle:
            data1 = pickle.load(handle)

        print('data1', data1)
        df = torch.from_numpy(np.array(data1['body_pose']))
        dt = torch.reshape(df, (1, 24, 3))
        db = dt[:,1:22,:] #for real time model
        #dt = torch.reshape(df, (1, 23, 3)) # for simulation model
        #db = dt[:,:21,:] # for simulation model offline model check this
        self.m_pose = dt
        orient_tensor = torch.from_numpy(np.array(data1['global_orient']))
        self.orient_body = orient_tensor.numpy()
        
        print('-------orient_body-----', self.orient_body*180/np.pi)
        #print('-------bodyshape---',dt)
        body_pose = np.zeros((1,23*3))
        self.smpl_body_pose = db[0].numpy()
        self.human_pos_offset, self.human_orient_offset = self.human.get_base_pos_orient()
        self.henry_joints = np.array(data1['human_joints_3D_est'])+np.array([-0.35,-1.3,1.2])

        orient_tensor = torch.from_numpy(np.array(data1['global_orient']))
        self.orient_body = orient_tensor.numpy()
        ang = self.orient_body[2]
        self.human.set_base_pos_orient([0, 0.02, 0.99195], [-np.pi/2.0, ang, 0])  # p.getQuaternionFromEuler(np.array(euler))
        #print('---henry_joints-----',self.henry_joints)
        #output = model(betas=torch.Tensor(data1['betas']), body_pose=data1['body_pose'], return_verts=True)
        #output = model(betas=torch.Tensor(data1['betas']), body_pose=torch.Tensor(body_pose), return_verts=True)

        # for i in range(len(self.henry_joints)):
        #     self.create_sphere(radius=0.01, mass=0.0, pos=self.henry_joints[i], visual=True, collision=False, rgba=[1, 0, 0, 1])

        # rh_s = np.array(self.human.get_pos_orient(self.human.right_shoulder)[0] ) + np.array([0,0,0.1])
        # rh_e = np.array(self.human.get_pos_orient(self.human.right_elbow)[0] ) + np.array([0,0,0.1])
        # rh_w = np.array(self.human.get_pos_orient(self.human.right_wrist)[0] ) + np.array([0,0,0.1])
        
        # lh_s = np.array(self.human.get_pos_orient(self.human.left_shoulder)[0] ) + np.array([0,0,0.1])
        # lh_e = np.array(self.human.get_pos_orient(self.human.left_elbow)[0] ) + np.array([0,0,0.1])
        # lh_w = np.array(self.human.get_pos_orient(self.human.left_wrist)[0] ) + np.array([0,0,0.1])

        # self.create_sphere(radius=0.01, mass=0.0, pos=rh_s, visual=True, collision=False, rgba=[0, 1, 0, 1])
        # self.create_sphere(radius=0.01, mass=0.0, pos=rh_e, visual=True, collision=False, rgba=[0, 1, 0, 1])
        # self.create_sphere(radius=0.01, mass=0.0, pos=rh_w, visual=True, collision=False, rgba=[0, 1, 0, 1])
        # self.create_sphere(radius=0.01, mass=0.0, pos=lh_s, visual=True, collision=False, rgba=[0, 1, 0, 1])
        # self.create_sphere(radius=0.01, mass=0.0, pos=lh_e, visual=True, collision=False, rgba=[0, 1, 0, 1])
        # self.create_sphere(radius=0.01, mass=0.0, pos=lh_w, visual=True, collision=False, rgba=[0, 1, 0, 1])



        #joints = output.joints.detach().cpu().numpy().squeeze()
        #print('output joint', joints.shape, joints)
        #return joints
            
        #output = model(betas=torch.Tensor(data1['betas']), body_pose=db, return_verts=True)
            
##--------------------------------------
# -- Joint Legend --

# assistive-gym-joints
        # self.j_right_pecs_x, self.j_right_pecs_y, self.j_right_pecs_z = 0, 1, 2
        # self.j_right_shoulder_x, self.j_right_shoulder_y, self.j_right_shoulder_z = 3, 4, 5
        # self.j_right_elbow = 6
        # self.j_right_forearm = 7
        # self.j_right_wrist_x, self.j_right_wrist_y = 8, 9

        # self.j_left_pecs_x, self.j_left_pecs_y, self.j_left_pecs_z = 10, 11, 12
        # self.j_left_shoulder_x, self.j_left_shoulder_y, self.j_left_shoulder_z = 13, 14, 15
        # self.j_left_elbow = 16
        # self.j_left_forearm = 17
        # self.j_left_wrist_x, self.j_left_wrist_y = 18, 19

        # self.j_neck = 20
        # self.j_head_x, self.j_head_y, self.j_head_z = 21, 22, 23
        # self.j_stomach = 24
        # self.j_waist_x, self.j_waist_y, self.j_waist_z = 25, 26, 27
        
        # self.j_right_hip_x, self.j_right_hip_y, self.j_right_hip_z = 28, 29, 30
        # self.j_right_knee = 31
        # self.j_right_ankle_x, self.j_right_ankle_y, self.j_right_ankle_z = 32, 33, 34
        # self.j_left_hip_x, self.j_left_hip_y, self.j_left_hip_z = 35, 36, 37
        # self.j_left_knee = 38
        # self.j_left_ankle_x, self.j_left_ankle_y, self.j_left_ankle_z = 39, 40, 41





# amass human model
# (0, b'lhip', 
# (1, b'lknee'
# (2, b'lankle',
# (3, b'rhip', 
# (4, b'rknee', 
# (5, b'rankle', 
# (6, b'lowerback'.
# (7, b'upperback'0
# (8, b'chest', 
# (9, b'lowerneck'
# (10, b'upperneck 
# (11, b'lclavicle0,
# (12, b'lshoulder.
# (13, b'lelbow', 
# (14, b'lwrist', 
# (15, b'rclavicle
# (16, b'rshoulder
# (17, b'relbow', 
# (18, b'rwrist', 


# given the amass human model all joint angles, we need to map the joint angles to the assistive-gym human model

# joint_names = [self.human.right_x, ]



#corresponding_smpl_orders = [14, 17, 19, 21, 21, 13, 16, 18, 20, 20, 12, 15, 3, 0, 2, 5, 8, 1, 4, 7]

    def convert_smpl_body_to_gym_new_not_working(self):
        smpl_bp_old = self.smpl_body_pose
    
        #R.from_rotvec(pr).as_quat()

        smpl_bp = np.zeros((22,3))
        for i in range(len(smpl_bp_old)):
            smpl_bp[i] = R.from_rotvec(smpl_bp_old[i]).as_euler('xyz') 
        # right shoulder 13
        opts_joints = [0,1,2,3,4,5,6]
        cor_angles = [smpl_bp[13,0],smpl_bp[13,1],smpl_bp[13,2],-smpl_bp[16,0],-smpl_bp[16,1],smpl_bp[16,2],smpl_bp[18,0] ] 
        self.human.set_joint_angles(opts_joints, cor_angles)
        
        #left shoulder 12
        opts_joints = [10,11,12,13,14,15,16]
        cor_angles = [smpl_bp[12,0],smpl_bp[12,1],smpl_bp[12,2],-smpl_bp[15,0],-smpl_bp[15,1],smpl_bp[15,2],smpl_bp[17,0] ] 
        self.human.set_joint_angles(opts_joints, cor_angles)

        #right leg
        opts_joints = [28,29,30,31,32,33,34]
        cor_angles = [smpl_bp[1,0],smpl_bp[1,1],smpl_bp[1,2],smpl_bp[4,0],smpl_bp[4,1],smpl_bp[4,2],smpl_bp[7,0]]
        self.human.set_joint_angles(opts_joints, cor_angles)
        print('right leg cor_angles', np.array(cor_angles)*180/3.14)
        #left leg
        opts_joints = [35,36,37,38,39,40,41]
        cor_angles = [-smpl_bp[0,0],smpl_bp[0,1],smpl_bp[0,2],smpl_bp[3,0],smpl_bp[3,1],smpl_bp[3,2],smpl_bp[6,0]]
        self.human.set_joint_angles(opts_joints, cor_angles)
        print('left leg cor_angles', np.array(cor_angles)*180/3.14)
        # head
        opts_joints = [20, 21,22,23]
        cor_angles = [smpl_bp[11,0],smpl_bp[14,0],smpl_bp[14,1],smpl_bp[14,2]]
        self.human.set_joint_angles(opts_joints, cor_angles)

        # waist
        opts_joints = [24,25,26,27]
        cor_angles = [smpl_bp[2,0],smpl_bp[0,0],smpl_bp[0,1],smpl_bp[0,2]] 
        self.human.set_joint_angles(opts_joints, cor_angles)


#self.henry_joint

#for 
#ik_angles_body =  p.calculateInverseKinematics(self.body, target_joint, targetPosition=target_pos)

#ref: https://files.is.tue.mpg.de/black/talks/SMPL-made-simple-FAQs.pdf



            
    def convert_smpl_body_to_gym(self):

        smpl_pose_jt_1 = self.smpl_body_pose  #self.smpl_agym_map(self.smpl_body_pose)
        
        print('orient_body', self.orient_body)
        #smpl_pose_jt_1 = smpl_pose_jt_1.dot(R.from_euler('x', -90, degrees=True).as_matrix())
        smpl_pose_jt_1 = smpl_pose_jt_1.dot(R.from_quat(self.human_orient_offset).as_matrix())
        #smpl_pose_jt_1 = smpl_pose_jt_1.dot(R.from_euler('y', 180, degrees=True).as_matrix())
        #smpl_pose_jt_1 = smpl_pose_jt_1.dot(R.from_euler('z', 180, degrees=True).as_matrix())
        
        smpl_bp = smpl_pose_jt_1
        # joints_1 = smpl_pose_jt_1.dot(R.from_euler('x', -90, degrees=True).as_matrix())
        # smpl_pose_jt_2 = joints_1.dot(R.from_quat(self.human_orient_offset).as_matrix())
        # smpl_pose_jt_2 = smpl_pose_jt_2.dot(R.from_euler('y', 180, degrees=True).as_matrix())
        # smpl_pose_jt = smpl_pose_jt_2.dot(R.from_euler('z', 180, degrees=True).as_matrix())

        

        self.human.set_base_pos_orient([0, 0.0, 0.80], [-np.pi/2, self.orient_body[2],0])

        # opts_joints = [ self.human.j_head_x, self.human.j_head_y, self.human.j_head_z,self.human.j_neck,
        #                 self.human.j_upper_chest_x, self.human.j_upper_chest_y, self.human.j_upper_chest_z,
        #                 self.human.j_chest_x, self.human.j_chest_y, self.human.j_chest_z,
        #                 self.human.j_waist_x, self.human.j_waist_y, self.human.j_waist_z ]

        # cor_angles = [  smpl_bp[14,0],smpl_bp[14,1],smpl_bp[14,2],smpl_bp[11,0],
        #                 smpl_bp[8,0],smpl_bp[8,1],smpl_bp[8,2],
        #                 smpl_bp[5,0],smpl_bp[5,1],smpl_bp[5,2],
        #                 smpl_bp[2,0],smpl_bp[2,1],smpl_bp[2,2] ]

        # self.human.set_joint_angles(opts_joints, cor_angles)
        #print('stomach cor_angles', np.array(cor_angles)*180/3.14)
        

        opts_joints = [ self.human.j_left_hip_x, self.human.j_left_hip_y, self.human.j_left_hip_z, self.human.j_left_knee ,
                        self.human.j_left_ankle_x, self.human.j_left_ankle_y, self.human.j_left_ankle_z ]

        cor_angles = [smpl_bp[0,0],smpl_bp[0,1],smpl_bp[0,2],smpl_bp[3,0],
                      smpl_bp[6,0],smpl_bp[6,1],smpl_bp[6,2]]

        self.human.set_joint_angles(opts_joints, cor_angles)
        #self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=4.8)
        #print('right leg cor_angles', np.array(cor_angles)*180/3.14)

        opts_joints = [ self.human.j_right_hip_x, self.human.j_right_hip_y, self.human.j_right_hip_z, self.human.j_right_knee, 
                        self.human.j_right_ankle_x, self.human.j_right_ankle_y, self.human.j_right_ankle_z]

        cor_angles = [smpl_bp[1,0],smpl_bp[1,1],smpl_bp[1,2],smpl_bp[4,0],
                      smpl_bp[7,0],smpl_bp[7,1],smpl_bp[7,2]]

        self.human.set_joint_angles(opts_joints, cor_angles)
        #self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=4.8)
        #print('left leg cor_angles', np.array(cor_angles)*180/3.14)

        opts_joints = [ self.human.j_left_pecs_x, self.human.j_left_pecs_y, self.human.j_left_pecs_z ,
                        self.human.j_left_shoulder_x, self.human.j_left_shoulder_y, self.human.j_left_shoulder_z ,
                        self.human.j_left_elbow, self.human.j_left_forearm ]
        

        cor_angles_left = [  -smpl_bp[12,2]-1.57,smpl_bp[12,1],smpl_bp[12,0],
                             ((-smpl_bp[15,2])-1.57),smpl_bp[15,1],-(smpl_bp[15,0]),
                             smpl_bp[17,2],smpl_bp[19,1]]

        #cor_angles_left = [0,0,0,-3.14,0,0,0,0]
        #shoulder x didn't make any difference

        self.human.set_joint_angles(opts_joints, cor_angles_left)
        #self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=4.8)
        #print('left hand cor_angles', np.array(cor_angles_left)*180/3.14)

        opts_joints = [ self.human.j_right_pecs_x, self.human.j_right_pecs_y, self.human.j_right_pecs_z ,
                        self.human.j_right_shoulder_x, self.human.j_right_shoulder_y, self.human.j_right_shoulder_z,
                        self.human.j_right_elbow, self.human.j_right_forearm]
        

        
        ck = smpl_bp[16,0] if smpl_bp[16,0]<0 else smpl_bp[16,0]+1.57

        cor_angles_right = [smpl_bp[13,2],smpl_bp[13,1],smpl_bp[13,0],
                            1.57-smpl_bp[16,2],smpl_bp[16,1],ck,
                            -smpl_bp[18,2],smpl_bp[20,1]]


        self.human.set_joint_angles(opts_joints, cor_angles_right)
        #self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=4.8)
        print('right hand cor_angles', np.array(cor_angles_right)*180/3.14) 

    def get_human_joint_position(self):

        agym_jt = np.zeros((20,3))
        #agym_jt
        agym_jt[0,:] = self.human.get_pos_orient(self.human.head)[0]
        agym_jt[1,:] = self.human.get_pos_orient(self.human.j_neck)[0] # try lower_neck
        agym_jt[2,:] = self.human.get_pos_orient(self.human.stomach)[0]
        agym_jt[3,:] = self.human.get_pos_orient(self.human.waist)[0]
        #agym_jt[4,:] = self.human.get_pos_orient(self.human.upper_chest)[0]
        #agym_jt[5,:] = self.human.get_pos_orient(self.human.j_upper_chest_x)[0]
        
        # arms - adding this updated version of the arms but leaving the original in case anything goes wrong
        agym_jt[6,:] = self.human.get_pos_orient(self.human.right_shoulder)[0]
        agym_jt[7,:] = self.human.get_pos_orient(self.human.left_shoulder)[0]
        agym_jt[8,:] = self.human.get_pos_orient(self.human.right_elbow)[0]
        agym_jt[9,:] = self.human.get_pos_orient(self.human.left_elbow)[0]
        agym_jt[10,:] = self.human.get_pos_orient(self.human.right_wrist)[0] 
        agym_jt[11,:] = self.human.get_pos_orient(self.human.left_wrist)[0]
        agym_jt[12,:] = self.human.get_pos_orient(self.human.right_pecs)[0]
        agym_jt[13,:] = self.human.get_pos_orient(self.human.left_pecs)[0]
        
        # 7,9,11,13 left arm
        # 6,8,10,12 right arm
        #legs - same with arms
        agym_jt[14,:] = self.human.get_pos_orient(self.human.right_knee)[0]
        agym_jt[15,:] = self.human.get_pos_orient(self.human.left_knee)[0]
        agym_jt[16,:] = self.human.get_pos_orient(self.human.right_hip)[0]
        agym_jt[17,:] = self.human.get_pos_orient(self.human.left_hip)[0]
        agym_jt[18,:] = self.human.get_pos_orient(self.human.right_ankle)[0]
        agym_jt[19,:] = self.human.get_pos_orient(self.human.left_ankle)[0]

        # for i_ in range(20):
        #     agym_jt[i_][2] = agym_jt[i_][2]+0.3

        return agym_jt



    def optimize_to_joints(self):
        
        lh_s = np.array(self.human.get_pos_orient(self.human.left_shoulder)[0] ) + np.array([0,0,0.1])
        lh_e = np.array(self.human.get_pos_orient(self.human.left_elbow)[0] ) + np.array([0,0,0.1])
        lh_w = np.array(self.human.get_pos_orient(self.human.left_wrist)[0] ) + np.array([0,0,0.1])
    
        hm_to_smpl = np.array([0,2,5,11,1,4,10,3,6,12,15,16,18,20,17,19,21])
        flh_s = self.henry_joints[hm_to_smpl[11]] + np.array([0,0,0.1])
        flh_e = self.henry_joints[hm_to_smpl[12]] + np.array([0,0,0.1])
        flh_w = self.henry_joints[hm_to_smpl[13]] + np.array([0,0,0.1])
        self.create_sphere(radius=0.01, mass=0.0, pos=flh_s, visual=True, collision=False, rgba=[0, 0, 1, 1])
        self.create_sphere(radius=0.01, mass=0.0, pos=flh_e, visual=True, collision=False, rgba=[0, 0, 1, 1])
        self.create_sphere(radius=0.01, mass=0.0, pos=flh_w, visual=True, collision=False, rgba=[0, 0, 1, 1])
        
        left_hand_error = np.linalg.norm(flh_s-lh_s) + np.linalg.norm(flh_e-lh_e) + np.linalg.norm(flh_w-lh_w)

        return left_hand_error
    
        # ass_gym_joints = self.get_human_joint_position()
        # error_jts_2d = np.zeros((20,3))

        # error_sum = 0
        # for i_ in range(20):
        #     if i_ in [6,8,10,12]:
        #         error_jts_2d[i_][0] = henry_joints[i_][0]-ass_gym_joints[i_][0]
        #         error_jts_2d[i_][1] = henry_joints[i_][1]-ass_gym_joints[i_][1]
        #         error_jts_2d[i_][2] = henry_joints[i_][2]-ass_gym_joints[i_][2]
        #         error_sum = error_sum + np.sqrt(error_jts_2d[i_][0]**2 + error_jts_2d[i_][1]**2 + error_jts_2d[i_][2]**2 )

        # frh_s = self.henry_joints[hm_to_smpl[14]] + np.array([0,0,0.1])
        # frh_e = self.henry_joints[hm_to_smpl[15]] + np.array([0,0,0.1])
        # frh_w = self.henry_joints[hm_to_smpl[16]] + np.array([0,0,0.1])
        # self.create_sphere(radius=0.01, mass=0.0, pos=frh_s, visual=True, collision=False, rgba=[0, 0, 1, 1])
        # self.create_sphere(radius=0.01, mass=0.0, pos=frh_e, visual=True, collision=False, rgba=[0, 0, 1, 1])
        # self.create_sphere(radius=0.01, mass=0.0, pos=frh_w, visual=True, collision=False, rgba=[0, 0, 1, 1])

        # right_hand_error = np.norm(frh_s-rh_s) + np.norm(frh_e-rh_e) + np.norm(frh_w-rh_w)

        # return error_sum
    

    def optimization_human(self, point):
        opts_joints =  [self.human.j_left_shoulder_x, self.human.j_left_shoulder_y, self.human.j_left_shoulder_z,self.human.j_left_elbow ]

        self.human.set_joint_angles(opts_joints, point)
        p.stepSimulation(physicsClientId=self.id)

        f_value = self.optimize_to_joints()

        #print(' cost is ', f_value)
        return f_value



    def bound_offset(self, opts_joints):
        #opts_joints = [self.human.j_left_shoulder_x, self.human.j_left_shoulder_y, self.human.j_left_shoulder_z,self.human.j_left_elbow ]
        bnd_list = []
        for i_ in opts_joints:
            bnd_list = bnd_list + [[self.human.lower_limits[i_],self.human.upper_limits[i_]] ]
        return bnd_list
    


    def cma_optimizer(self, opts_joints):

        ln = len(opts_joints)
        x0 = np.zeros([ln])
        bnd = self.bound_offset(opts_joints)
        opts = cma.CMAOptions({'verb_disp': 1, 'popsize': 8})
        opts.set('bounds',bnd)
        opts.set('tolfun', 6e-2)
        opts['tolx'] = e-2

        cma_option = {"BoundaryHandler": cma.BoundTransform,"bounds": bnd,}

        opts.update(cma_option)

        es = cma.CMAEvolutionStrategy(x0, 0.2)
        logger = cma.CMADataLogger().register(es)
        #es.optimize(optimizing_function, args=(env_human, test_set))
        es.optimize(self.optimization_human,iterations=10, opts=opts, callback=es.logger.plot)
        #es.result_pretty()
        print('---result---',es.result.xbest)
        cma.plot()
        logger.plot()
        print('FInished')
        return es.result.xbest


    def scipy_optimizer(self, henry_joints):
        x0 = np.zeros([10])
        bnd = self.bound_offset()
        opt = {'disp':True,'maxiter':4000} 
        result = optimize.minimize(fun = self.optimization_human, x0=x0, method='Nelder-Mead', bounds=bnd ,options=opt )
        print(' cost is ', result.fun, ' and solution is ', result.x)
        print('---------------------------------------------------------------------------')
        return result.fun, result.x


    def pose_reindex(self):

        smpl_pose_jt_1 =  self.load_smpl_model()
        self.human_pos_offset, self.human_orient_offset = self.human.get_base_pos_orient()

        joints_1 = smpl_pose_jt_1.dot(R.from_euler('x', -90, degrees=True).as_matrix())
        smpl_pose_jt_2 = joints_1.dot(R.from_quat(self.human_orient_offset).as_matrix())
        smpl_pose_jt_2 = smpl_pose_jt_2.dot(R.from_euler('y', 180, degrees=True).as_matrix())
        smpl_pose_jt = smpl_pose_jt_2.dot(R.from_euler('z', 180, degrees=True).as_matrix())
        return self.smpl_agym_map(smpl_pose_jt)


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


