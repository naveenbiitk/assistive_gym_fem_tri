import gym, assistive_gym, argparse
import pybullet as p
import numpy as np

import os
from assistive_gym.learn import make_env
from math import sin,cos
from PIL import Image
import pickle
import cma
import yaml

best_params = []
best_costs = []
fevals = 0
f = open("all_cmaes_data_handover","wb")


def calculate_human_Jacobian(env_human, env_id):
    #joints_nd = env_human.right_arm_joints
    joints_nd = env_human.body
    joints_nd = range(p.getNumJoints(env_human.body) )
    joint_states = p.getJointStates(env_human.body, joints_nd, physicsClientId=env_id)
    joint_infos = [p.getJointInfo(env_human.body, i, physicsClientId= env_id) for i in (joints_nd)]
    motor_states = [j for j, i in zip(joint_states, joint_infos) if i[2] != p.JOINT_FIXED]
    motor_indices = [i[0] for j, i in zip(joint_states, joint_infos) if i[2] != p.JOINT_FIXED]
    motor_positions = [state[0] for state in motor_states]
    motor_velocities = [state[1] for state in motor_states]
    motor_torques = [state[3] for state in motor_states]
    #return motor_indices, motor_positions, motor_velocities, motor_torques

    ee = env_human.right_wrist
    joint_velocities = [0.0] * len(motor_positions)
    joint_accelerations = [0.0] * len(motor_positions)
    center_of_mass = p.getLinkState(env_human.body, ee, computeLinkVelocity=True, computeForwardKinematics=True, physicsClientId=env_id)[2]
    #print('motor_positions',motor_positions)
    #print('motor_velocities',joint_velocities)
    #print('Center_of_mass',center_of_mass)
    J_linear, J_angular = p.calculateJacobian(env_human.body, ee, localPosition=center_of_mass, objPositions=motor_positions, objVelocities=joint_velocities, objAccelerations=joint_accelerations, physicsClientId=env_id)

    J_linear = np.array(J_linear)[:, env_human.right_arm_joints]
    J_angular = np.array(J_angular)[:, env_human.right_arm_joints]
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

                   
def calculate_human_Jlwki(env_human, env_id, target_pose_angles):
    human_right_arm_lower_limits = [env_human.lower_limits[i] for i in env_human.right_arm_joints]
    human_right_arm_upper_limits = [env_human.upper_limits[i] for i in env_human.right_arm_joints]

    lower_limits = human_right_arm_lower_limits
    upper_limits = human_right_arm_upper_limits

    J = calculate_human_Jacobian(env_human, env_id)
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



def calculate_obj_ik(env, target_pose_angles, target_position):

  mid_angle = np.array([-1.28484584e-14,  1.74532925e-01,  1.24455499e-14, -8.98027383e-02, -1.10828408e+00,  
              6.82478992e-14, -1.11701072e+00, -1.43938821e-14,  7.85398163e-02,  1.74532925e-01])
  
  max_moment = 363.4084744
  max_ang = 16.7190253038

  weights_opt_ik = np.array([ 0.5,  0.5,  0.1 ])

  forces = np.array([env.human.get_force_torque_sensor(joints_j) for joints_j in env.human.right_arm_joints])

  linear_forces = np.sqrt( forces[:,0]*forces[:,0] + forces[:,1]*forces[:,1] + forces[:,2]*forces[:,2] )
  linear_moment = np.sqrt( forces[:,3]*forces[:,3] + forces[:,4]*forces[:,4] + forces[:,5]*forces[:,5] )
  
  f_moment = np.sum(linear_moment**2)
  f_moment = f_moment/(max_moment**2)

  f_angle = np.sum( (target_pose_angles-mid_angle)**2 )
  f_angle = f_angle/(max_ang**2)

  current_position = env.human.get_pos_orient(env.human.right_wrist)[0]
  f_position = np.linalg.norm(target_position-current_position)

  f_final = (f_moment*weights_opt_ik[0] + f_angle*weights_opt_ik[1] + f_position*weights_opt_ik[2])/(np.sum(weights_opt_ik))
  
  return f_final



def position_human_arm(env, target_position, attempts):
  
    human_angles = env.human.get_joint_angles(env.human.right_arm_joints)
    iteration = 0

    best_sol = None
    best_joint_positions = None

    while iteration < attempts:
        iteration += 1
        target_joint_angles_h = env.human.ik(env.human.right_wrist, target_position, None, ik_indices=env.human.right_arm_joints, max_iterations=2000)

        env.human.set_joint_angles(env.human.right_arm_joints, target_joint_angles_h)

        f_sol = calculate_obj_ik(env, target_joint_angles_h, target_position)
        #print('f_sol----',f_sol)
        if best_sol is None or f_sol < best_sol:
            best_sol = f_sol
            best_joint_positions = target_joint_angles_h

    return best_joint_positions
                


def run_simulation(env_human, env_id, count_target, env, target_position):

    env.target_pos = np.array(target_position)
    env.target = env.create_sphere(radius=0.02, mass=0.0, pos= target_position, visual=True, collision=False, rgba=[1, 1, 0, 1]) 

    target_joint_angles_h = position_human_arm(env, target_position, attempts=50)
    count_ = 0
    #env_human.set_joint_angles(env_human.right_arm_joints, target_joint_angles_h)
    while count_<count_target:
        
        #env.render()
        # p.removeUserDebugItem(t, env.id)
        # print('--Joint angles', target_pose_angles)
        # print(env.human.right_arm_joints)
        current_joint_angles = env.human.get_joint_angles(env.human.right_arm_joints)
        kp=0.6
        set_p = current_joint_angles + kp*(target_joint_angles_h - current_joint_angles)
        env_human.set_joint_angles(env_human.right_arm_joints, set_p)
        # if coop:
        #    action ={'robot': env.action_space_robot.sample(), 'human': env.action_space_human.sample()}
        human_action = np.zeros([8])
        observation, reward, done, info = env.step(human_action)

        count_=count_+1

    return target_joint_angles_h

 



def optimizing_function(points, env_human, env_id, env):
    
    global best_params, best_costs, fevals, f, dict_file

    mid_angle = np.array([-1.28484584e-14,  1.74532925e-01,  1.24455499e-14, -8.98027383e-02, -1.10828408e+00,  
                6.82478992e-14, -1.11701072e+00, -1.43938821e-14,  7.85398163e-02,  1.74532925e-01])

    weights_opt = np.array([ 0.5,  0.5,  0.05 ])
    #(f_moment f_angle  f_manipulability)

    dist_threshold=0.1
    max_moment = 363.4084744
    max_ang = 16.7190253038
    max_det = 0.0314498201


    target_position = np.array(points)

    target_pose_angles_h = run_simulation(env_human, env_id ,18, env, target_position)
    
    env_human.set_joint_angles(env_human.right_arm_joints, target_pose_angles_h)
    
    forces = np.array([env_human.get_force_torque_sensor(joints_j) for joints_j in env_human.controllable_joint_indices])
    
    current_endpose, current_orient = env_human.get_pos_orient(env_human.right_wrist)
    
    #print('Current end pose', current_endpose)
    distance = np.linalg.norm(current_endpose-target_position)

    det, jlwki = calculate_human_Jlwki(env_human, env_id, target_pose_angles_h)
    det = det/max_det

    if distance < dist_threshold:
        det=1/det
        f_manipulability = det #1-det
        #print('------------Reached----------------')
    else:
        det=0.00000000
        f_manipulability = 50
        #print('Not Reached')

    
    linear_forces = np.sqrt( forces[:,0]*forces[:,0] + forces[:,1]*forces[:,1] + forces[:,2]*forces[:,2] )
    linear_moment = np.sqrt( forces[:,3]*forces[:,3] + forces[:,4]*forces[:,4] + forces[:,5]*forces[:,5] )
    
    
    f_moment = np.sum(forces[:7,5]**2)/5
    f_moment = f_moment  #/(max_moment)

    f_angle = np.sum( (target_pose_angles_h-mid_angle)**2 )
    f_angle = f_angle/(max_ang**2)

    #manipulability, _ = calculate_human_manipulability(env_human, env_id, target_pose_angles)
    #manipulability = manipulability/max_manipulability
    #f_manipulability = 1-manipulability
    #f_manipulability = 1-det

    f_value = (weights_opt[0]*f_moment + weights_opt[1]*f_angle + weights_opt[2]*f_manipulability)/(weights_opt[0]+weights_opt[1]+weights_opt[2])
    

    return f_value



def optimizer_cma(env):
    
    wrist_pos,wrist_orient = env.human.get_pos_orient(env.human.right_wrist)
    shoulder_pos,shoulder_orient = env.human.get_pos_orient(env.human.right_shoulder)
    arm_length = np.linalg.norm(wrist_pos-shoulder_pos)-0.1
    arm_length = 0.1
    sh_pos, sh_orient = env.human.get_pos_orient(env.human.right_shoulder)

    #x0 = [ 0.05079058, 0.04465512, 0.08518684, 0.08726646, 0.00902719, -0.01202248,  0.0,  -0.00942944,  -0.3599609,  0.82030475]
    x0 = [-0.1, -0.1, -0.3]

    opts = cma.CMAOptions({'verb_disp': 1, 'popsize': 8})
    opts.set('bounds',[ [-(arm_length-0.1),-arm_length,-(arm_length-0.1) ], [(arm_length-0.1),0,(arm_length-0.1) ] ])

    opts.set('tolfun', 6e-2)
    opts['tolx'] = 6e-2

    cma_option = {"BoundaryHandler": cma.BoundTransform,
                "bounds": [ [-(arm_length-0.1),-arm_length,-(arm_length-0.1) ], [(arm_length-0.1),0,(arm_length-0.1) ] ],}

    opts.update(cma_option)

    es = cma.CMAEvolutionStrategy(x0, 0.2)
    logger = cma.CMADataLogger().register(es)
    #es.optimize(optimizing_function, args=(env_human, test_set))
    es.optimize(optimizing_function,args=(env.human, env.id), opts=opts, callback=es.logger.plot)
    es.result_pretty()
    cma.plot()
    logger.plot()
    print('FInished')

    with open(r'cmaes_handover_store_file.yaml', 'w') as file:
        documents = yaml.dump(dict_file, file)



if __name__ == "__main__":
    env_name = "ObjectHandoverStretchHuman-v1"
    env = make_env(env_name, coop=True)
    # env = gym.make()
    env.render()
    observation = env.reset()
    base_pos_set = [5,-5,0.1]
    quat_orient = env.get_quaternion([0.0, 0.0, 3.14])
    env.robot.set_base_pos_orient( base_pos_set, quat_orient)

    #p.removeBody(1 , physicsClientId=env.id)
    #test_set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # right arm shoulder and elbow
    #optimizer(env)
    points = [ -0.322725220284914, -0.25785464645505627, 0.00928032059021836]
    #points = [0.18, -0.4, 0.1]  #best one
    #points = [-0.08, -0.25, -0.25]
    optimizing_function(points, env.human, env.id, env)