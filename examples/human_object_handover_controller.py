import gym, assistive_gym, argparse
import pybullet as p
import numpy as np

import os
import ray._private.utils
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter

from assistive_gym.learn import make_env
from math import sin,cos
from PIL import Image
import pickle
import cma


env_name = "ObjectHandoverStretchHuman-v1"
env = make_env(env_name, coop=True)
# env = gym.make()
env.render()
observation = env.reset()
#observation= np.concatenate((observation['robot'], observation['human']))
#env.robot.print_joint_info()


# Arrow keys for moving the base, s/x for the lift, z/c for the prismatic joint, a/d for the wrist joint
#robot_actions = {p.B3G_LEFT_ARROW: np.array([0.01, -0.01, 0, 0, 0]), p.B3G_RIGHT_ARROW: np.array([-0.01, 0.01, 0, 0, 0]), p.B3G_UP_ARROW: np.array([0.01, 0.01, 0, 0, 0]), p.B3G_DOWN_ARROW: np.array([-0.01, -0.01, 0, 0, 0]), ord('s'): np.array([0, 0, 0.01, 0, 0]), ord('x'): np.array([0, 0, -0.01, 0, 0]), ord('z'): np.array([0, 0, 0, -0.01, 0]), ord('c'): np.array([0, 0, 0, 0.01, 0]), ord('a'): np.array([0, 0, 0, 0, 0.01]), ord('d'): np.array([0, 0, 0, 0, -0.01])}

human_actions = { ord('1'): np.array([ 0.01, 0, 0, 0]), ord('2'): np.array([ -0.01, 0, 0, 0]), ord('3'): np.array([ 0, 0.01, 0, 0]), ord('4'): np.array([ 0, -0.01, 0, 0]), ord('5'): np.array([ 0, 0, -0.01, 0]), ord('6'): np.array([ 0, 0, 0.01, 0]), ord('7'): np.array([ 0, 0, 0, 0.01]), ord('8'): np.array([0, 0, 0, -0.01])}

# start
right_arm_angles_start = np.array([0.0,  0.0,  5.31665415e-01,  1.34692067e-01, -9.64689935e-02, -2.42673015e-01, -1.64185327e+00,  0.0,  0.0,  0.0])

#end
# right_arm_angles_end = np.array([0.0,  0.0, 0.0,  0.6285238,  -1.2184946,   0.0,  -2.2070327,  0.0, 0.0, 0.0])
right_arm_angles_end = np.array([0.0,  0.0, 0.0,  0.6285238 - 0.5,  -1.2184946,   0.0 + 0.25,  -2.2070327,  0.0, 0.0, 0.0])

delta = (right_arm_angles_end - right_arm_angles_start) / 100



keys_actions = {p.B3G_LEFT_ARROW: np.array([0.01]), p.B3G_RIGHT_ARROW: np.array([-0.01]) }




#mouth_pos = [0, -0.11, 0.03] #if human.gender == 'male' else [0, -0.1, 0.03]
mouth_pos = [0, -0.11, 0.1]
head_pos, head_orient = env.human.get_pos_orient(env.human.head)
target_pos, target_orient = p.multiplyTransforms(head_pos, head_orient, mouth_pos, [0, 0, 0, 1], physicsClientId=env.id)

print('Target pose ', target_pos)
#target_s = env.create_sphere(radius=0.01, mass=0.0, pos=target_pos, visual=True, collision=False, rgba=[0, 1, 1, 1])

#target_pose_angles = env.human.ik(env.human.right_wrist, target_pos, target_orient, ik_indices=env.human.right_arm_joints, max_iterations=2000)
target_pose_angles = np.array([ 0.16973331, 0.2782036, -0.51687032, 1.02134001,-2.23546802, 0.19530055,-0.08927307,-1.2609814, 1.25120709, -0.27932173])
target_pose_angles = np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
i = [0, 1, 2, 3] #angle 4.1 max left up --- angle 2.1 max right up
#i = [3] #max up
#i = [1] #right slight
#i = [4] #angle -2.1  max forward
#i = [3, 6] #angle -2.8  good handover position
#i = [1, 4] #angle 4.8 perfect back
#i = [1, 2, 4]  #angle -2.2,2.2 perfect right
#i = [2, 4] # angle 1.2 perfect bottom right
#i = [1,2,4]

target_pose_angles[i] = 4.1
# jacobians ?
env.human.set_joint_angles(env.human.right_arm_joints, target_pose_angles)


def calculate_human_Jacobian(env):
    #joints_nd = env.human.right_arm_joints
    joints_nd = env.human.body
    joints_nd = range(p.getNumJoints(env.human.body) )
    joint_states = p.getJointStates(env.human.body, joints_nd, physicsClientId=env.id)
    joint_infos = [p.getJointInfo(env.human.body, i, physicsClientId= env.id) for i in (joints_nd)]
    motor_states = [j for j, i in zip(joint_states, joint_infos) if i[2] != p.JOINT_FIXED]
    motor_indices = [i[0] for j, i in zip(joint_states, joint_infos) if i[2] != p.JOINT_FIXED]
    motor_positions = [state[0] for state in motor_states]
    motor_velocities = [state[1] for state in motor_states]
    motor_torques = [state[3] for state in motor_states]
    #return motor_indices, motor_positions, motor_velocities, motor_torques

    ee = env.human.right_wrist
    joint_velocities = [0.0] * len(motor_positions)
    joint_accelerations = [0.0] * len(motor_positions)
    center_of_mass = p.getLinkState(env.human.body, ee, computeLinkVelocity=True, computeForwardKinematics=True, physicsClientId=env.id)[2]
    #print('motor_positions',motor_positions)
    #print('motor_velocities',joint_velocities)
    #print('Center_of_mass',center_of_mass)
    J_linear, J_angular = p.calculateJacobian(env.human.body, ee, localPosition=center_of_mass, objPositions=motor_positions, objVelocities=joint_velocities, objAccelerations=joint_accelerations, physicsClientId=env.id)

    J_linear = np.array(J_linear)[:, env.human.right_arm_joints]
    J_angular = np.array(J_angular)[:, env.human.right_arm_joints]
    #print('Jacobian linear size', np.shape(J_linear) )
    #print('Jacobian rotation size', np.shape(J_angular) )
    J = np.concatenate([J_linear, J_angular], axis=0)
    return J


def joint_limited_weighting(q, lower_limits, upper_limits):
    phi = 0.5
    lam = 0.05
    weights = []
    for qi, l, u in zip(q, lower_limits, upper_limits):
        qr = 0.5*(u - l)
        weights.append(1.0 - np.power(phi, (qr - np.abs(qr - qi + l)) / (lam*qr) + 1))
        if weights[-1] < 0.001:
            weights[-1] = 0.001
        # Joint-limited-weighting
    joint_limit_weight = np.diag(weights)
    return joint_limit_weight

                   
def calculate_human_Jlwki(env, target_pose_angles):
    human_right_arm_lower_limits = [env.human.lower_limits[i] for i in env.human.right_arm_joints]
    human_right_arm_upper_limits = [env.human.upper_limits[i] for i in env.human.right_arm_joints]

    lower_limits = human_right_arm_lower_limits
    upper_limits = human_right_arm_upper_limits

    J = calculate_human_Jacobian(env)
    joint_positions_q_star = target_pose_angles
    joint_limit_weight = joint_limited_weighting(joint_positions_q_star, lower_limits, upper_limits)
    #print('joint_limit_weight', joint_limit_weight)
    joint_limit_weight[0,0]=0.99
    # Joint-limited-weighted kinematic isotropy (JLWKI)
    det = max(np.linalg.det(np.matmul(np.matmul(J, joint_limit_weight), J.T)), 0)
    #print('determinant_value',det*10000)
    a=6 #order of robot space 6D (3D position and 3D orientation)
    jlwki = np.power(det, 1.0/a)/(np.trace(np.matmul(np.matmul(J, joint_limit_weight), J.T))/a)
    return det,jlwki

# def take_picture(renderer, width=256, height=256, scale=0.001, conn_id=None):
#     view_matrix = p.computeViewMatrix([0, 0, -1], [0, 0, 0], [0, -1, 0], physicsClientId=conn_id)
#     proj_matrix = p.computeProjectionMatrixFOV(20, 1, 0.05, 2, physicsClientId=conn_id)
#     w, h, rgba, depth, mask = p.getCameraImage(width=width,height=height,projectionMatrix=proj_matrix,viewMatrix=view_matrix,renderer=renderer,physicsClientId=conn_id)
#     return rgba 
# print('Jacobian',J)

print('Target pose angles ', target_pose_angles)

print('jlwki ', calculate_human_Jlwki(env, target_pose_angles) )

wrist_pos,wrist_orient = env.human.get_pos_orient(env.human.right_wrist)
#env.create_sphere(radius=0.01, mass=0.0, pos=wrist_pos, visual=True, collision=False, rgba=[0, 1, 1, 1])

shoulder_pos,shoulder_orient = env.human.get_pos_orient(env.human.right_shoulder)
# env.create_sphere(radius=0.1, mass=0.0, pos=wrist_pos, visual=True, collision=False, rgba=[0, 1, 1, 1])
arm_length = np.linalg.norm(wrist_pos-shoulder_pos)+0.2
print('Arm arm_length-------', arm_length)

# target_pose = env.target_pos
# target_pose_angles = env.robot.ik(env.robot.left_end_effector, human_target, orient, ik_indices=env.robot.left_arm_ik_indices, max_iterations=200)
# print('robot target angle',target_pose_angles)
# current_joint_angles = env.robot.get_joint_angles(env.robot.left_arm_joint_indices)
# print('current_joint_angles of robot',current_joint_angles)
# beta = count_/200
# action = beta*target_pose_angles + (1.0-beta)*right_arm_angles_start

action = np.zeros(10)
right_side = False
count_ = 0

reward_total = 0

i_ = 0



radi_max = 8
theta_max = 8
phi_max = 4

points_sphere = np.zeros([ radi_max*theta_max*phi_max ,3])
check_mask = np.zeros([ radi_max*theta_max*phi_max ])

for r in range(2,radi_max):
    r=r*arm_length/radi_max
    for theta in range(0,theta_max):
        theta = theta*2*np.pi/theta_max
        for phi in range(0,phi_max):            
            phi = phi*np.pi/phi_max
            x,y,z=  r*cos(theta)*sin(phi), r*sin(theta)*sin(phi), r*cos(phi)  
            
            #if phi > 2*np.pi/3:
                #env.create_sphere(radius=0.01, mass=0.0, pos=points_sphere[i_], visual=True, collision=False, rgba=[0, 1, 1, 1])   
            #else:
                #env.create_sphere(radius=0.01, mass=0.0, pos=points_sphere[i_], visual=True, collision=False, rgba=[0, 1, 1, 1])
            #if theta == 6*2*np.pi/theta_max and phi== 2*np.pi/phi_max:
            if theta != 600 :            
                points_sphere[i_] = shoulder_pos + np.array([x,y,z])
                #points_sphere[i_] =  np.array([x,y,z])
                print('Points',x,y,z)
                env.create_sphere(radius=0.01, mass=0.0, pos=points_sphere[i_], visual=True, collision=False, rgba=[0, 1, 1, 1])
                i_ = i_+1


# print(shoulder_pos)
# print('points min and max',np.min(points_sphere[:,0]), np.max(points_sphere[:,0]) )
# print('points min and max',np.min(points_sphere[:,1]), np.max(points_sphere[:,1]) )
# print('points min and max',np.min(points_sphere[:,2]), np.max(points_sphere[:,2]) )

# with open('file_handover_best.pkl', 'rb') as f:
#     loaded_dict = pickle.load(f)

# det_list = np.array(loaded_dict['determinant'])
# det_list_points = np.array(loaded_dict['determinant_points'])
# sforce_list = np.array(loaded_dict['sum_force']) #192*10
# sforce_list_points = np.array(loaded_dict['sum_force_points'])
# smoment_list = np.array(loaded_dict['sum_moment']) #192*10
# smoment_list_points = np.array(loaded_dict['sum_moment_points'])



def generate_line(pos, orient, lineLen=0.5):
        
    #p.removeAllUserDebugItems()
    mat = p.getMatrixFromQuaternion(orient)
    dir0 = [mat[0], mat[3], mat[6]]
    dir1 = [mat[1], mat[4], mat[7]]
    dir2 = [mat[2], mat[5], mat[8]]
    
    # works only for hand 0.25 linelen
    #dir2_neg = [-mat[2], -mat[5], -mat[8]]
    #to1 = [pos[0] + lineLen * (dir2_neg[0]+dir0[0])/2, pos[1] + lineLen * (dir2_neg[1]+dir0[1])/2, pos[2] + lineLen * (dir2_neg[2]+dir0[2])/2 ]
    #to2 = [pos[0] + lineLen * (dir2_neg[0]-dir0[0])/2, pos[1] + lineLen * (dir2_neg[1]-dir0[1])/2, pos[2] + lineLen * (dir2_neg[2]+dir0[2])/2 ]
    
    # works only for head  1.5 linlen
    dir2_neg = [-mat[1], -mat[4], -mat[7]]
    to1 = [pos[0] + lineLen * (dir2_neg[0]+dir0[0])/2, pos[1] + lineLen * (dir2_neg[1]+dir0[1])/2, pos[2] + lineLen * (dir2_neg[2]+dir0[2])/2 ]
    to2 = [pos[0] + lineLen * (dir2_neg[0]-dir0[0])/2, pos[1] + lineLen * (dir2_neg[1]-dir0[1])/2, pos[2] + lineLen * (dir2_neg[2]+dir0[2])/2 ]
    
    toX = [pos[0] + lineLen * dir0[0], pos[1] + lineLen * dir0[1], pos[2] + lineLen * dir0[2]]
    toY = [pos[0] + lineLen * dir1[0], pos[1] + lineLen * dir1[1], pos[2] + lineLen * dir1[2]]
    toZ = [pos[0] + lineLen * dir2[0], pos[1] + lineLen * dir2[1], pos[2] + lineLen * dir2[2]]
    
    p.addUserDebugLine(pos, toX, [1, 0, 0], 5)
    p.addUserDebugLine(pos, toY, [0, 1, 0], 5)
    p.addUserDebugLine(pos, toZ, [0, 0, 1], 5)

    #p.addUserDebugLine(pos, to1, [0, 1, 1], 5, 3)
    #p.addUserDebugLine(pos, to2, [0, 1, 1], 5, 3)
    #p.addUserDebugLine(to2, to1, [0, 1, 1], 5, 3)



def optimizing_function(points):
    
    target_position = root_pos + np.array(points)

    # if np.linalg.norm(target_position)==0:
    #     continue

    target_pose_angles = env.human.ik(env.human.right_wrist, target_position, None, ik_indices=env.human.right_arm_joints, max_iterations=2000)
    print('Iteration ',sim_)

    target_ee_pos, target_ee_orient = env.human.get_pos_orient(env.human.head)
    
    robot_base_arrange(env, target_position)
    run_simulation(env, target_pose_angles , 15)
    env.human.set_joint_angles(env.human.right_arm_joints, target_pose_angles)
    forces = np.array([env.human.get_force_torque_sensor(joints_j) for joints_j in env.human.controllable_joint_indices])
    
    current_endpose, current_orient = env.human.get_pos_orient(env.human.right_wrist)
    distance = np.linalg.norm(current_endpose-target_position)

    if distance < dist_threshold:
        check_mask[sim_]=1
        print('Reached')
    else:
        check_mask[sim_]=0
        print('Not Reached')

    #det, jlwki = calculate_human_Jlwki(env, target_pose_angles)
    
    linear_forces = np.sqrt( forces[:,0]*forces[:,0] + forces[:,1]*forces[:,1] + forces[:,2]*forces[:,2] )
    linear_moment = np.sqrt( forces[:,3]*forces[:,3] + forces[:,4]*forces[:,4] + forces[:,5]*forces[:,5] )
    
    f_moment = np.sum(linear_moment**2)
    f_angle = np.sum( (target_pose_angles-mid_angle)**2 )

    f_value = weights_opt[0]*f_moment + weights_opt[1]*f_angle + weights_opt[2]*f_jacobian 
    return f_value




env.setup_camera_rpy(camera_target=[-0.2, 0, 0.75], distance=1.5, rpy=[0, -35, 40], fov=60, camera_width=1920//4, camera_height=1080//4)


target_ee_pos = np.array([-0.6, 0, 0.8]) 
#target_ee_orient = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task])
target_ee_orient = np.array([0, 0, 0, 1 ])

#env.init_robot_pose(target_ee_pos, target_ee_orient, [(target_ee_pos, target_ee_orient), (target_position, None)], [(target_position, target_ee_orient)], arm='left', tools=[env.tool], collision_objects=[env.human, env.furniture])
#self.init_robot_pose(target_ee_pos, target_ee_orient, [(target_ee_pos, target_ee_orient), (self.target_pos, None)], [(self.target_pos, target_ee_orient)], arm='right', tools=None, collision_objects=[self.human, self.furniture])

   
# Continually randomize the robot base position and orientation
# Continually randomize the robot base position and orientation

# env.robot.position_robot_toc(task, arms, start_pos_orient, target_pos_orients, human, base_euler_orient=np.zeros(3), max_ik_iterations=200, max_ik_random_restarts=1, randomize_limits=False, attempts=100, jlwki_restarts=1, step_sim=False, check_env_collisions=False, right_side=True, random_rotation=30, random_position=0.5):
# Continually randomize the robot base position and orientation
# Continually randomize the robot base position and orientation

#env.init_robot_pose(target_ee_pos, target_ee_orient, start_pos_orient, target_pos_orients, arm='right', tools=[], collision_objects=[], wheelchair_enabled=True, right_side=True, max_iterations=3):
def multi_joint_optimization(joint_angles, human, joint_indices):
    env.human.set_joint_angles(joint_indices, joint_angles)
    lower_limits = np.array([human.lower_limits[ind] for ind in joint_indices])
    upper_limits = np.array([human.upper_limits[ind] for ind in joint_indices])
    joint_ranges = upper_limits - lower_limits
    joint_centers = lower_limits + (joint_ranges / 2.)
    current_joint_angles = np.array(human.get_joint_angles(joint_indices))
    distance_from_center = current_joint_angles - joint_centers
    distance_from_center_normalized = distance_from_center / joint_ranges
    penalty = sum(abs(distance_from_center_normalized))
    return penalty



def robot_base_arrange(env, target_pos):
    base_pos, base_orient = env.robot.get_pos_orient(env.robot.base)
    #pf = generate_target_point(base_pos,base_orient)
    #print('Robot base orient', base_orient)
    pf = target_pos
    yaw_orientation = np.arctan2(2*(base_orient[3]*base_orient[2]+base_orient[0]*base_orient[1]),1-2*(base_orient[1]*base_orient[1]+base_orient[2]*base_orient[2]))
    #base_pos_set = [pf[0]-0.0*cos(yaw_orientation)-0.3,pf[1]-0.0*sin(yaw_orientation)-0.7,base_pos[2]]
    base_pos_set = [pf[0]-0.0*cos(yaw_orientation)-1.8,pf[1]-0.0*sin(yaw_orientation)-1.5,base_pos[2]]
    y2= base_pos_set[1]-target_pos[1]
    x2= base_pos_set[0]-target_pos[0]
    yaw_sp = np.arctan2(x2,y2) - np.pi/2
    quat_orient = env.get_quaternion([0.0, 0.0, yaw_sp])
    env.robot.set_base_pos_orient( base_pos_set, quat_orient)


def run_simulation(env, target_pose_angles, count_target):
    coop = True
    count_=0
    t=0
    while count_<count_target:
        
        # env.render()
        # p.removeUserDebugItem(t, env.id)
        env.human.set_joint_angles(env.human.right_arm_joints, target_pose_angles)
        det, jlwki= calculate_human_Jlwki(env, target_pose_angles)
        st = "%.3f" % jlwki
        # t = p.addUserDebugText(text=st, textPosition=[-0.8,0,1], textColorRGB=[1,1,1], textSize=2, lifeTime=0.5, physicsClientId=env.id )

        if coop:
            action ={'robot': env.action_space_robot.sample(), 'human': env.action_space_human.sample()}

        observation, reward, done, info = env.step(action)
        count_=count_+1

        if count_==5:
            img, depth = env.get_camera_image_depth()
            im = Image.fromarray(img)
            #im.save(st+'.png')



best_jlwki = 0.28
jlwki = 0
det_list = []
jlwki_list = []
forces_list = []
sforce_list = []
smoment_list = []
f_moment_list = []
f_angle_list = []

# distance curent_endpose-setpoint distance>thresold mask = 0
# current_endpose current_pos, current_orient = env.human.get_pos_orient(env.human.right_wrist)
# setpoint = target_position = points_sphere[sim_]

# for sim_ in range(i_):
#     target_position = points_sphere[sim_]


# make midangle check objective function
# current_angle    mid_angle = []

# f_angle = np.sum( (target_pose_angles-mid_angle)**2 ) 



mid_angle = np.array([-1.28484584e-14,  1.74532925e-01,  1.24455499e-14, -8.98027383e-02, -1.10828408e+00,  
                6.82478992e-14, -1.11701072e+00, -1.43938821e-14,  7.85398163e-02,  1.74532925e-01])

dist_threshold=0.1
# for sim_ in range(radi_max*theta_max*phi_max):
for sim_ in range(i_):
    target_position = points_sphere[sim_]

    if np.linalg.norm(target_position)==0:
        continue

    target_pose_angles = env.human.ik(env.human.right_wrist, target_position, None, ik_indices=env.human.right_arm_joints, max_iterations=2000)
    print('Iteration ',sim_)

    target_ee_pos, target_ee_orient = env.human.get_pos_orient(env.human.head)
    #env.init_robot_pose(target_ee_pos, target_ee_orient, [(target_ee_pos, target_ee_orient), (target_position, None)], [(target_position, target_ee_orient)], arm='left', tools=[env.tool], collision_objects=[env.human, env.furniture])
    robot_base_arrange(env, target_position)
    run_simulation(env, target_pose_angles , 15)
    env.human.set_joint_angles(env.human.right_arm_joints, target_pose_angles)
    forces = np.array([env.human.get_force_torque_sensor(joints_j) for joints_j in env.human.controllable_joint_indices])
    
    current_endpose, current_orient = env.human.get_pos_orient(env.human.right_wrist)
    distance = np.linalg.norm(current_endpose-target_position)

    if distance < dist_threshold:
        check_mask[sim_]=1
        print('Reached')
    else:
        check_mask[sim_]=0
        print('Not Reached')

    #det, jlwki = calculate_human_Jlwki(env, target_pose_angles)
    
    linear_forces = np.sqrt( forces[:,0]*forces[:,0] + forces[:,1]*forces[:,1] + forces[:,2]*forces[:,2] )
    linear_moment = np.sqrt( forces[:,3]*forces[:,3] + forces[:,4]*forces[:,4] + forces[:,5]*forces[:,5] )
    
    f_moment = np.sum(linear_moment**2)
    f_angle = np.sum( (target_pose_angles-mid_angle)**2 )

    #det_list.append(det)
    #jlwki_list.append(jlwki)
    #forces_list.append(forces)
    sforce_list.append(linear_forces)
    smoment_list.append(linear_moment)
    f_moment_list.append(f_moment)
    f_angle_list.append(f_angle)
 
    # print('Forces ', np.max(linear_forces), 'Moments ', f_moment,'Shoulder Moments ', linear_moment[5] )
    # print('Joint angles', env.human.get_joint_angles([3,4,5]) )
    # print('Mid angle', f_angle)
    # print('------------------------------')
    # if jlwki > best_jlwki:
    #     print('------------------This is the best-------------')
    #     best_jlwki = jlwki
    #     best_joint_positions = target_pose_angles
    #     run_simulation(env, target_pose_angles , 25)
    # print('jlwki ', jlwki )

#target_pose_angles = best_joint_positions
# target_joint_angles
# f_moment
# f_angle
# check_mask

dict_item = { 'Points': points_sphere,'sum_force': sforce_list, 'sum_moment': smoment_list,'reach_check': check_mask,'fobj_moment':f_moment_list, 'fobj_angle':f_angle_list }
f = open("file_handover_2.pkl","wb")
pickle.dump(dict_item,f)
f.close()


test_set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # right arm shoulder and elbow
#test_set = get_articulated_joint_indices(env.human)
#x0 = env.human.get_joint_angles(test_set)
x0 = [ 0.05079058, 0.04465512, 0.08518684, 0.08726646, 0.00902719, -0.01202248,  0.0,  -0.00942944,  -0.3599609,  0.82030475]

es = cma.CMAEvolutionStrategy(x0, 0.001)
es.optimize(multi_joint_optimization, args=(env.human, test_set))
es.result_pretty()


count_=0
while count_<52:
    env.render()
    # action = np.zeros(env.action_robot_len)
        # if count_<50:
        #     base_position, _, _ = env.robot.position_robot_toc(env.task, 'left', pos, target_pose, env.human, step_sim=False, check_env_collisions=True, max_ik_iterations=100, max_ik_random_restarts=1, randomize_limits=False, right_side=right_side, base_euler_orient=[0, 0, 0 if right_side else np.pi], attempts=50)
        # print('inside loop')
    current_joint_angles = env.human.get_joint_angles(env.human.right_arm_joints)

    generate_line([-0.5,0,1], [0,0,0,1], lineLen=0.5)
    env.human.set_joint_angles(env.human.right_arm_joints, target_pose_angles*0)
    # env.human.set_joint_angles(env.human.right_arm_joints, right_arm_angles_end)

    #kp=0.5
    #if count_>0:
    #    err_angles = target_pose_angles-current_joint_angles
    #else:
    #    err_angles = right_arm_angles_end - current_joint_angles
    
    #action = kp*(err_angles)
    
    # print('current joint angles',current_joint_angles)

    #current_joint_angles = env.robot.get_joint_angles(env.robot.left_gripper_indices)
    #action[len(env.robot.wheel_joint_indices):] = target_pose_angles-current_joint_angles
            
    keys = p.getKeyboardEvents()
    count_ = count_ + 1
    #print("actions", np.linalg.norm(action),"count",count_)

    #action = right_arm_angles_start
    #print(env.robot.get_joint_angles(env.robot.left_arm_joint_indices))
    for key, a in keys_actions.items():
        if key in keys and keys[key] & p.KEY_IS_DOWN:
            action_ += a
            #action = action*100

    action_ = dict()
    action_['robot'] = env.action_space_robot.sample() # Get a random action
    action_['human'] = env.action_space_human.sample() # Get a random action
    #wrist_pos,wrist_orient = env.human.get_pos_orient(env.human.right_wrist)        
    #env.create_sphere(radius=0.01, mass=0.0, pos=wrist_pos, visual=True, collision=False, rgba=[0, 1, 1, 1])
    observation, reward, done, info = env.step(action_)
    #reward_total += reward
    #task_success = info['task_success']
    count_ = count_+1


#print('reward')
#print(reward_total)
print(es.result)
print('Current joint angles', current_joint_angles )
print('task_success')
#print(task_success)








