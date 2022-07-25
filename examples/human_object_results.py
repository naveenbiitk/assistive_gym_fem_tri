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
import time



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
    return det*10000,jlwki



def robot_base_arrange(env, target_pos):
    base_pos, base_orient = env.robot.get_pos_orient(env.robot.base)
    #pf = generate_target_point(base_pos,base_orient)
    #print('Robot base orient', base_orient)
    pf = target_pos
    yaw_orientation = np.arctan2(2*(base_orient[3]*base_orient[2]+base_orient[0]*base_orient[1]),1-2*(base_orient[1]*base_orient[1]+base_orient[2]*base_orient[2]))
    base_pos_set = [pf[0]-0.0*cos(yaw_orientation)-0.3,pf[1]-0.0*sin(yaw_orientation)-0.5,base_pos[2]]
    #base_pos_set = [pf[0]-0.0*cos(yaw_orientation)-1.8,pf[1]-0.0*sin(yaw_orientation)-1.5,base_pos[2]]
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
        
        #env.render()
        #p.removeUserDebugItem(t, env.id)
        env.human.set_joint_angles(env.human.right_arm_joints, target_pose_angles)
        det, jlwki= calculate_human_Jlwki(env, target_pose_angles)
        st = "%.3f" % jlwki
        #t = p.addUserDebugText(text=st, textPosition=[-0.8,0,1], textColorRGB=[1,1,1], textSize=2, lifeTime=0.5, physicsClientId=env.id )

        
        if coop:
            action ={'robot': env.action_space_robot.sample(), 'human': env.action_space_human.sample()}

        observation, reward, done, info = env.step(action)
        count_=count_+1

        if count_==5:
            img, depth = env.get_camera_image_depth()
            im = Image.fromarray(img)
            #im.save(st+'.png')



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
        #toY = [pos[0] - lineLen * dir1[0], pos[1] - lineLen * dir1[1], pos[2] - lineLen * dir1[2]]
        
        p.addUserDebugLine(pos, toX, [1, 0, 0], 5)
        p.addUserDebugLine(pos, toY, [0, 1, 0], 5)
        p.addUserDebugLine(pos, toZ, [0, 0, 1], 5)


def generate_line_head(pos, orient, lineLen=0.5):
        
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
    
    #p.addUserDebugLine(pos, toX, [1, 0, 0], 5)
    toY = [pos[0] - lineLen * dir1[0], pos[1] - lineLen * dir1[1], pos[2] - lineLen * dir1[2]]
    p.addUserDebugLine(pos, toY, [1, 0, 0], 5)
    #p.addUserDebugLine(pos, toZ, [0, 0, 1], 5)
    return toY

env_name = "ObjectHandoverStretchHuman-v1"
env = make_env(env_name, coop=True)
#env = gym.make()
env.render()
observation = env.reset()
#observation= np.concatenate((observation['robot'], observation['human']))
#env.robot.print_joint_info()


shoulder_pos,shoulder_orient = env.human.get_pos_orient(env.human.right_shoulder)
wrist_pos,wrist_orient = env.human.get_pos_orient(env.human.right_wrist)
# env.create_sphere(radius=0.1, mass=0.0, pos=wrist_pos, visual=True, collision=False, rgba=[0, 1, 1, 1])
arm_length = np.linalg.norm(wrist_pos-shoulder_pos)+0.2
print('Arm arm_length-------', arm_length)


env.setup_camera_rpy(camera_target=[-0.2, 0, 0.75], distance=1.5, rpy=[0, -35, 40], fov=60, camera_width=1920//4, camera_height=1080//4)

target_ee_pos = np.array([-0.6, 0, 0.8]) 
#target_ee_orient = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task])
target_ee_orient = np.array([0, 0, 0, 1 ])

count_=0

robot_pos = [-0.59404564, -0.66383235 , 0.78296778]
robot_base_arrange(env, robot_pos)


shoulder_pos,shoulder_orient = env.human.get_pos_orient(env.human.right_shoulder)


head_t_pos, head_t_orient = env.human.get_pos_orient(env.human.head)
eye_pos = [0, -0.11, 0.1] if env.human.gender == 'male' else [0, -0.1, 0.1] # orientation needed is [0, 0, 0.707, -0.707]
    # right_ear_pos = [-0.08, -0.011, 0.08] if self.human.gender == 'male' else [-0.08, -0.01, 0.08]
heye_pos, heye_orient = p.multiplyTransforms(head_t_pos, head_t_orient, eye_pos, [0, 0, 0, 1], physicsClientId=env.id)



#points = [-0.04050275653090826, -0.38547224068225105, -0.14691517168799585]

#points = [-0.1693589450778777, -0.3614343639495855, 0.16958656076588924]
#points = [ 0.00126, -0.31976, -0.08800]
points = [0.18, -0.25, 0.1]
#target_pos = [-0.29404564, -0.46383235 , 0.78296778]
target_pos = [points[0]+shoulder_pos[0], points[1]+shoulder_pos[1], points[2]+shoulder_pos[2]  ]

env.create_sphere(radius=0.02, mass=0.0, pos= target_pos, visual=True, collision=False, rgba=[1, 1, 0, 1]) 
#generate_line(target_pos, wrist_orient, lineLen=0.1)

#target_position = [-0.29404564, -0.26383235,  0.78296778]
#target_position = [-0.63320341, -0.68258234,  0.71888622]

target_position = target_pos
#robot_base_arrange(env, target_pos)


#for _ in range(50):
#    p.stepSimulation(physicsClientId=env.id)

#time.sleep(20)

while count_<200:
    env.render()
    generate_line_head(heye_pos, heye_orient)
    target_pose_angles = env.human.ik(env.human.right_wrist, target_position, None, ik_indices=env.human.right_arm_joints, max_iterations=5000)

    current_joint_angles = env.human.get_joint_angles(env.human.right_arm_joints)

    env.human.set_joint_angles(env.human.right_arm_joints, target_pose_angles)

    run_simulation(env, target_pose_angles, 100)

             
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


