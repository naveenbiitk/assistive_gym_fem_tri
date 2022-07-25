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


env_name = "HumanTesting-v1"
env = make_env(env_name, coop=True)
#env = gym.make()
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
target_s = env.create_sphere(radius=0.01, mass=0.0, pos=target_pos, visual=True, collision=False, rgba=[0, 1, 1, 1])

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

    # Joint-limited-weighted kinematic isotropy (JLWKI)
    det = max(np.linalg.det(np.matmul(np.matmul(J, joint_limit_weight), J.T)), 0)
    a=6 #order of robot space 6D (3D position and 3D orientation)
    jlwki = np.power(det, 1.0/a)/(np.trace(np.matmul(np.matmul(J, joint_limit_weight), J.T))/a)
    return jlwki

# def take_picture(renderer, width=256, height=256, scale=0.001, conn_id=None):
#     view_matrix = p.computeViewMatrix([0, 0, -1], [0, 0, 0], [0, -1, 0], physicsClientId=conn_id)
#     proj_matrix = p.computeProjectionMatrixFOV(20, 1, 0.05, 2, physicsClientId=conn_id)
#     w, h, rgba, depth, mask = p.getCameraImage(width=width,height=height,projectionMatrix=proj_matrix,viewMatrix=view_matrix,renderer=renderer,physicsClientId=conn_id)
#     return rgba 
# print('Jacobian',J)

print('Target pose angles ', target_pose_angles)

print('jlwki ', calculate_human_Jlwki(env, target_pose_angles) )

wrist_pos,wrist_orient = env.human.get_pos_orient(env.human.right_wrist)
env.create_sphere(radius=0.01, mass=0.0, pos=wrist_pos, visual=True, collision=False, rgba=[0, 1, 1, 1])

shoulder_pos,shoulder_orient = env.human.get_pos_orient(env.human.right_shoulder)
# env.create_sphere(radius=0.1, mass=0.0, pos=wrist_pos, visual=True, collision=False, rgba=[0, 1, 1, 1])
arm_length = np.linalg.norm(wrist_pos-shoulder_pos)
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
points_sphere = np.zeros([5*5*3,3])


for r in range(0,5):
    r=r*arm_length/5+0.1
    for theta in range(0,5):
        theta = theta*2*np.pi/5
        for phi in range(0,3):            
            phi = phi*np.pi/3

            if theta == 1*2*np.pi/5 and phi== np.pi/3:
                x,y,z=  r*cos(theta)*sin(phi), r*sin(theta)*sin(phi), r*cos(phi)  
                points_sphere[i_] = shoulder_pos + np.array([x,y,z])
                env.create_sphere(radius=0.01, mass=0.0, pos=points_sphere[i_], visual=True, collision=False, rgba=[0, 1, 1, 1])   
                i_ = i_+1


env.setup_camera_rpy(camera_target=[-0.2, 0, 0.75], distance=1.5, rpy=[0, -35, 40], fov=60, camera_width=1920//4, camera_height=1080//4)


def run_simulation(env, target_pose_angles, count_target):
    count_=0
    t=0
    st = "0.0"
    while count_<count_target:
        
        env.render()
        p.removeUserDebugItem(t, env.id)
        env.human.set_joint_angles(env.human.right_arm_joints, target_pose_angles)
        
        observation, reward, done, info = env.step(action)
        count_=count_+1

        if count_==5:
            img, depth = env.get_camera_image_depth()
            im = Image.fromarray(img)
            forces = np.array([env.human.get_force_torque_sensor(joints_j) for joints_j in env.human.controllable_joint_indices])

            linear_forces = np.sqrt(forces[:,0]*forces[:,0] + forces[:,1]*forces[:,1] + forces[:,2]*forces[:,2] )
            linear_moment = np.sqrt(forces[:,3]*forces[:,3] + forces[:,4]*forces[:,4] + forces[:,5]*forces[:,5] )


            jlwki = calculate_human_Jlwki(env, target_pose_angles)
            ml = 1000*np.max(linear_moment)
            st = "%.8f" % ml
            
            #im.save(st+'.png')
        t = p.addUserDebugText(text=st, textPosition=[-0.8,0,1], textColorRGB=[1,1,1], textSize=2, lifeTime=0.5, physicsClientId=env.id )




best_jlwki = 0.28
jlwki = 0

for sim_ in range(i_):
    target_position = points_sphere[sim_]
    target_pose_angles = env.human.ik(env.human.right_wrist, target_position, None, ik_indices=env.human.right_arm_joints, max_iterations=2000)
    print('Iteration ',sim_)
    run_simulation(env, target_pose_angles , 15)
    jlwki = calculate_human_Jlwki(env, target_pose_angles)


    if jlwki > best_jlwki:
        print('------------------This is the best-------------')
        best_jlwki = jlwki
        best_joint_positions = target_pose_angles
        run_simulation(env, target_pose_angles , 25)
    
    print('jlwki ', jlwki )

target_pose_angles = best_joint_positions

while count_<250:
    env.render()
    # action = np.zeros(env.action_robot_len)
        # if count_<50:
        #     base_position, _, _ = env.robot.position_robot_toc(env.task, 'left', pos, target_pose, env.human, step_sim=False, check_env_collisions=True, max_ik_iterations=100, max_ik_random_restarts=1, randomize_limits=False, right_side=right_side, base_euler_orient=[0, 0, 0 if right_side else np.pi], attempts=50)
        # print('inside loop')
    current_joint_angles = env.human.get_joint_angles(env.human.right_arm_joints)

    env.human.set_joint_angles(env.human.right_arm_joints, target_pose_angles)
    # env.human.set_joint_angles(env.human.right_arm_joints, right_arm_angles_end)

    kp=0.5
    if count_>0:
        err_angles = target_pose_angles-current_joint_angles
    else:
        err_angles = right_arm_angles_end - current_joint_angles
    
    action = kp*(err_angles)
    
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

    #wrist_pos,wrist_orient = env.human.get_pos_orient(env.human.right_wrist)        
    #env.create_sphere(radius=0.01, mass=0.0, pos=wrist_pos, visual=True, collision=False, rgba=[0, 1, 1, 1])
    observation, reward, done, info = env.step(action)
    reward_total += reward
    #task_success = info['task_success']


print('reward')
print(reward_total)
print('task_success')
#print(task_success)


