import gym, assistive_gym, argparse
import pybullet as p
import numpy as np

import os
from assistive_gym.learn import make_env
from math import sin,cos
from PIL import Image
import pickle
import cma
import math

best_params = []
best_costs = []
fevals = 0
f = open("all_cmaes_data_showphone","wb")


def calculate_human_Jacobian(env_human, env_id):
    #joints_nd = env_human.right_arm_joints
    #joints_nd = env_human.body
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


def find_nearest_point(A,C,B):
    # A = eye_pos
    # C = toY pose
    # B =  target_pos
    t = np.dot(B-A,C-A)/np.dot(C-A,C-A)
    D = A + t*(C-A)
    return D



def generate_line_head(pos, orient, lineLen=0.5):
    #p.removeAllUserDebugItems()
    mat = p.getMatrixFromQuaternion(orient)
    dir0 = [mat[0], mat[3], mat[6]]
    dir1 = [mat[1], mat[4], mat[7]]
    dir2 = [mat[2], mat[5], mat[8]]

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


def generate_line_axis(pos, orient, lineLen=0.10):
    #p.removeAllUserDebugItems()
    mat = p.getMatrixFromQuaternion(orient)
    dir0 = [mat[0], mat[3], mat[6]]
    dir1 = [mat[1], mat[4], mat[7]]
    dir2 = [mat[2], mat[5], mat[8]]

    # works only for head  1.5 linlen
    dir2_neg = [-mat[1], -mat[4], -mat[7]]
    to1 = [pos[0] + lineLen * (dir2_neg[0]+dir0[0])/2, pos[1] + lineLen * (dir2_neg[1]+dir0[1])/2, pos[2] + lineLen * (dir2_neg[2]+dir0[2])/2 ]
    to2 = [pos[0] + lineLen * (dir2_neg[0]-dir0[0])/2, pos[1] + lineLen * (dir2_neg[1]-dir0[1])/2, pos[2] + lineLen * (dir2_neg[2]+dir0[2])/2 ]
    
    toX = [pos[0] + lineLen * dir0[0], pos[1] + lineLen * dir0[1], pos[2] + lineLen * dir0[2]]
    toY = [pos[0] + lineLen * dir1[0], pos[1] + lineLen * dir1[1], pos[2] + lineLen * dir1[2]]
    toZ = [pos[0] + lineLen * dir2[0], pos[1] + lineLen * dir2[1], pos[2] + lineLen * dir2[2]]
    
    p.addUserDebugLine(pos, toX, [1, 0, 0], 5)
    #toY = [pos[0] - lineLen * dir1[0], pos[1] - lineLen * dir1[1], pos[2] - lineLen * dir1[2]]
    p.addUserDebugLine(pos, toY, [0, 1, 0], 5)
    p.addUserDebugLine(pos, toZ, [0, 0, 1], 5)
    return toY


def cosine_angle (a, b, c):
    return math.degrees(math.acos((c**2 - b**2 - a**2)/(-2.0 * a * b)))

def distance_points(a,b):
    return math.sqrt( (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2 )


def cone_points(p0, p1, p2):
    theta_check_threshold=30
    #p0 main origin
    #p1 linelength end axis  p2 check point
    theta = cosine_angle(distance_points(p0,p2),distance_points(p0,p1),distance_points(p1,p2) )
    print('----------Theta angle', theta)
    if theta < theta_check_threshold:
        point_inside_cone = True
    else:
        point_inside_cone = False

    return theta, point_inside_cone



def neck_angle(env_human, env_id, target_pos):

    eye_pos = [0, -0.11, 0.1] if env_human.gender == 'male' else [0, -0.1, 0.1]
    head_t_pos, head_t_orient = env_human.get_pos_orient(env_human.head)
    heye_pos, heye_orient = p.multiplyTransforms(head_t_pos, head_t_orient, eye_pos, [0, 0, 0, 1], physicsClientId=env_id)

    print('Distance---', np.linalg.norm(np.array(heye_pos)-np.array(target_pos)))
    p0 = heye_pos
    p1 = generate_line_head(heye_pos, heye_orient)
    p2 = target_pos
    
    l_p = find_nearest_point(p0,p1,p2)
    theta, is_inside = cone_points(p0, p1, p2)

    print('Neck theta', theta)
    return theta, is_inside



def run_simulation(env_human, env_id, target_pose_angles, count_target, env):
    coop = True
    count_=0
    t=0
    while count_<count_target:
        
        waist_pos, waist_orient = env_human.get_pos_orient(env_human.head)
        waist_pos_2 = [waist_pos[0], waist_pos[1]-0.15, waist_pos[2]]
        generate_line_axis(waist_pos_2, waist_orient)
        generate_line_head(waist_pos, waist_orient, lineLen=0.5)
        # waist_pos2, waist_orient2 = env_human.get_pos_orient(env_human.right_pecs)
        # waist_pos_22 = [waist_pos2[0], waist_pos2[1]-0.15, waist_pos2[2]]
        # generate_line_axis(waist_pos_22, waist_orient2)



        env.render()
        # p.removeUserDebugItem(t, env.id)

        #print('--Joint angles initial', target_pose_angles)
        
        #print('--Joint angles after', target_pose_angles)
        env.human.set_joint_angles(env.human.right_arm_joints, target_pose_angles,use_limits=False)
        det, jlwki= calculate_human_Jlwki(env_human, env_id, target_pose_angles)
        #st = "%.3f" % jlwki
        # t = p.addUserDebugText(text=st, textPosition=[-0.8,0,1], textColorRGB=[1,1,1], textSize=2, lifeTime=0.5, physicsClientId=env.id )

        if coop:
            action ={'robot': env.action_space_robot.sample(), 'human': env.action_space_human.sample()}

        human_action = np.zeros([4])
        observation, reward, done, info = env.step(action)
        count_ = count_ + 1

        #if count_==5:
        #    img, depth = env.get_camera_image_depth()
        #    im = Image.fromarray(img)
        #    im.save(st+'.png')



def optimizing_function(points, env_human, env_id, env, print_flag=False):
    
    global best_params, best_costs, fevals, f

    mid_angle = np.array([-1.28484584e-14,  1.74532925e-01,  1.24455499e-14, -8.98027383e-02, -1.10828408e+00,  
                6.82478992e-14, -1.11701072e+00, -1.43938821e-14,  7.85398163e-02,  1.74532925e-01])

    weights_opt = np.array([ 1,  1,  0.1 , 3])
    dist_threshold = 0.1
    max_moment = 363.4084744
    max_ang = 16.7190253038
    max_det = 0.0314498201
    neck_max = 30
    neck_med = 10

    root_pos, root_orient = env_human.get_pos_orient(env_human.right_shoulder)
    #shoulder_pos,shoulder_orient = env.human.get_pos_orient(env.human.right_shoulder)

    target_position = root_pos + np.array(points)
    #target_position = np.array(points)
    #print('Points', points)

    # if np.linalg.norm(target_position)==0:
    #     continue

    #p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0,1,1,1], physicsClientId=env_id)
    env.create_sphere(radius=0.02, mass=0.0, pos= target_position, visual=True, collision=False, rgba=[1, 1, 0, 1]) 
    target_pose_angles = env_human.ik(env_human.right_wrist, target_position, None, ik_indices=env_human.right_arm_joints, max_iterations=2000)
    #print('Iteration ',sim_)
    print('target angles ', target_pose_angles)
    target_ee_pos, target_ee_orient = env_human.get_pos_orient(env_human.head)
    
    #robot_base_arrange(env, target_position)
    env_human.set_joint_angles(env_human.right_arm_joints, target_pose_angles)
    run_simulation(env_human, env_id, target_pose_angles , 75, env)
    
    forces = np.array([env_human.get_force_torque_sensor(joints_j) for joints_j in env_human.controllable_joint_indices])
    
    current_endpose, current_orient = env_human.get_pos_orient(env_human.right_wrist)
    
    #print('Current end pose', current_endpose)
    distance = np.linalg.norm(current_endpose-target_position)

    det, jlwki = calculate_human_Jlwki(env_human, env_id, target_pose_angles)
    det = det/max_det

    if distance < dist_threshold:
        det=det
        f_manipulability = 1-det
        #print('------------Reached----------------')
    else:
        det=0.00000000
        f_manipulability = 20
        #print('Not Reached')

    theta_neck, is_inside = neck_angle(env_human, env_id, target_position)

    if is_inside:
        f_neck = (theta_neck - neck_med)**2
        f_neck = f_neck/(neck_max**2)
        #print('------------Reached----------------')
    else:
        f_neck = 50 
        #print('Not Reached')
    
    linear_forces = np.sqrt( forces[:,0]*forces[:,0] + forces[:,1]*forces[:,1] + forces[:,2]*forces[:,2] )
    linear_moment = np.sqrt( forces[:,3]*forces[:,3] + forces[:,4]*forces[:,4] + forces[:,5]*forces[:,5] )
    
    
    f_moment = np.sum(linear_moment**2)
    f_moment = f_moment/(max_moment**2)

    f_angle = np.sum( (target_pose_angles-mid_angle)**2 )
    f_angle = f_angle/(max_ang**2)

    #manipulability, _ = calculate_human_manipulability(env_human, env_id, target_pose_angles)
    #manipulability = manipulability/max_manipulability
    #f_manipulability = 1-manipulability
    #f_manipulability = 1-det

    f_value = (weights_opt[0]*f_moment + weights_opt[1]*f_angle + weights_opt[2]*f_manipulability + weights_opt[3]*f_neck)/(weights_opt[0]+weights_opt[1]+weights_opt[2]+ weights_opt[3])
    

    if print_flag:
        print('f_moment', f_moment)
        print('f_angle', f_angle)
        print('f_manipulability ', f_manipulability)
        print('f_neck', f_neck)
        print('F total', f_value)

    #print(f_value)

    # Keep track of the best 5 param sets
    # if len(best_costs) < 5 or f_value < best_costs[-1]:
    #     # Find location to insert new low cost param set
    #     index = 0
    #     for i in range(len(best_costs)):
    #         if f_value < best_costs[i]:
    #             index = i
    #             break
    #     best_params.insert(index, points)
    #     best_costs.insert(index, f_value)
    #     if len(best_costs) > 5:
    #         best_params.pop(-1)
    #         best_costs.pop(-1)
    # if fevals % 101 == 0:
    #     # if fevals % 50 == 0:
    #     # Print out best param sets so far
    #     print('Best Costs:', best_costs)
    #     print('Best Params:', [[('%.5f' % xx) for xx in x] for x in best_params])
    # fevals += 1
    # pickle.dump([fevals, points, f_value], f)

    return f_value


#[-0.009061834731738142, -0.09152791912650515, 0.4332791175015075]
#[0.04050275653090826, -0.38547224068225105, 0.012691517168799585]
#[-0.3429148673702284, -0.22968218368770588, -0.2086895573995843]
#[-0.1693589450778777, -0.3614343639495855, 0.16958656076588924]


def optimizer(env):
    
    wrist_pos,wrist_orient = env.human.get_pos_orient(env.human.right_wrist)
    shoulder_pos,shoulder_orient = env.human.get_pos_orient(env.human.right_shoulder)
    arm_length = np.linalg.norm(wrist_pos-shoulder_pos)-0.1

    x0 = [-0.2, -0.2, -0.3]

    es = cma.CMAEvolutionStrategy(x0, 0.20)
    opts = cma.CMAOptions({'verb_disp': 1, 'popsize': 8})
    opts.set('bounds',[ [-(arm_length-0.1),-arm_length,-(arm_length-0.1) ], [(arm_length-0.1),0,(arm_length-0.1) ] ])


    cma_option = {"BoundaryHandler": cma.BoundTransform,
            "bounds": [ [-(arm_length-0.1),-arm_length,-(arm_length-0.1) ], [(arm_length-0.1),0,(arm_length-0.1) ] ],
        }

    opts.update(cma_option)

    logger = cma.CMADataLogger().register(es)

    #es.optimize(optimizing_function, args=(env_human, test_set))
    es.optimize(optimizing_function,args=(env.human, env.id), opts=opts, callback=es.logger.plot)
    es.result_pretty()
    cma.plot()
    logger.plot()
    print('Finished')



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

    sh_pos, sh_orient = env.human.get_pos_orient(env.human.right_shoulder)

    #optimizer(env)

    points = [0.18, -0.4, 0.1]  #best one
    #points = [-0.08, -0.25, -0.25]
    optimizing_function(points, env.human, env.id, env, print_flag=True)

    #x0 = [ 0.05079058, 0.04465512, 0.08518684, 0.08726646, 0.00902719, -0.01202248,  0.0,  -0.00942944,  -0.3599609,  0.82030475]
    #{'boundary_handling': 'BoundPenalty','bounds': [ [-(arm_length-0.1),-arm_length,-(arm_length-0.1) ], [(arm_length-0.1),0,(arm_length-0.1) ] ] }



