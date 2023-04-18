import matplotlib.pyplot as plt
import pickle
import gym, assistive_gym, argparse
import pybullet as p
import numpy as np
import yaml
from cmaes import CMA

from assistive_gym.learn import make_env
from math import sin,cos
from PIL import Image
import math
import time
import pickle
from scipy import optimize
from scipy.optimize import Bounds
import os



def calculate_obj_ik(env, target_pose_angles, target_position):

  mid_angle = np.array([ 0.17401677,  0.04713208, -0.17548489,  0.977391,   -0.7791986,   1.25441974,
                -1.56912958, -1.0887199,   0.16446499,  0.75026786])
  
  max_moment = 363.4084744
  max_ang = 16.7190253038

  weights_opt_ik = np.array([ 0.5,  0.5,  0.1 ])

  forces = np.array([env.human.get_force_torque_sensor(joints_j) for joints_j in env.human.right_arm_joints])

  linear_forces = np.sqrt( forces[:,0]*forces[:,0] + forces[:,1]*forces[:,1] + forces[:,2]*forces[:,2] )
  linear_moment = np.sqrt( forces[:,3]*forces[:,3] + forces[:,4]*forces[:,4] + forces[:,5]*forces[:,5] )
  
  f_moment = np.sum(linear_moment**2)
  f_moment = f_moment/(max_moment)
  #f_moment = 0
  #f_moment = f_moment/(max_moment**2)

  #f_moment = np.sum(forces[:7,5]**2)/5
  #f_moment = f_moment  #/(max_moment)


  f_angle = np.sum( (target_pose_angles-mid_angle)**2 )
  f_angle = f_angle/(max_ang**2)

  current_position = env.human.get_pos_orient(env.human.right_wrist)[0]
  f_position = np.linalg.norm(target_position-current_position)*10

  if f_position>0.2:
    f_position = f_position*1000  
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
        p.stepSimulation(physicsClientId=env.id)
        #print('f_sol----',f_sol)
        if best_sol is None or f_sol < best_sol:
            best_sol = f_sol
            best_joint_positions = target_joint_angles_h
        p.stepSimulation(physicsClientId=env.id)
    #print('Best sol', best_sol)
    return best_joint_positions, best_sol


def point_wrt_body_frames(point, env, frame_i):

    head_point = np.array(env.human.get_pos_orient(env.human.head)[0],dtype="float64")
    neck_point = np.array(env.human.get_pos_orient(env.human.neck)[0],dtype="float64")
    chest_point = np.array(env.human.get_pos_orient(env.human.right_pecs)[0],dtype="float64")
    waist_point = np.array(env.human.get_pos_orient(env.human.waist)[0],dtype="float64")
    stomach_point = np.array(env.human.get_pos_orient(env.human.stomach)[0],dtype="float64")
    left_shoulder_point = np.array(env.human.get_pos_orient(env.human.left_shoulder)[0],dtype="float64")
    right_shoulder_point = np.array(env.human.get_pos_orient(env.human.right_shoulder)[0],dtype="float64")
    left_elbow_point = np.array(env.human.get_pos_orient(env.human.left_elbow)[0],dtype="float64")
    right_elbow_point = np.array(env.human.get_pos_orient(env.human.right_elbow)[0],dtype="float64")

    target_points = np.zeros([9,3])

    target_points[0] = head_point + point
    target_points[1] = neck_point + point
    target_points[2] = chest_point + point
    target_points[3] = waist_point + point
    target_points[4] = stomach_point + point
    target_points[5] = left_shoulder_point + point
    target_points[6] = right_shoulder_point + point
    target_points[7] = left_elbow_point + point
    target_points[8] = right_elbow_point + point

    return target_points[frame_i]




def optimizing_across_poses(point, env, frame_i):

    env.human.set_base_velocity(linear_velocity=[0, 0, 0],angular_velocity=[0, 0, 0])
    # observation = env.reset()
    human_base_pose, human_orient = env.human.get_pos_orient(env.human.base) 
    orientation = p.getEulerFromQuaternion(np.array(human_orient), physicsClientId=env.id) 
    index = 0
    # print('scipy point: ',point)
    evaluation_point = point_wrt_body_frames(point, env, frame_i)
    env.create_sphere(radius=0.02, mass=0.0, pos = evaluation_point, visual=True, collision=False, rgba=[1, 1, 0, 1])

    collision_detected = False
    collision_score = 0

    target_joint_angles_h, score = position_human_arm(env, evaluation_point, attempts=20)
    env.human.set_joint_angles(env.human.right_arm_joints, target_joint_angles_h)
    print('ja:',target_joint_angles_h)
    score_final = score+collision_score
    f_value = score_final*10000
    #print('Iteration done: f_value: ', f_value, point)
    return f_value                


def iterating_across_poses(point, env, frame_i):
    human_base_pose, human_orient = env.human.get_pos_orient(env.human.base) 
    orientation = p.getEulerFromQuaternion(np.array(human_orient), physicsClientId=env.id) 

    index = 0
    total_index=625
    score_array = np.zeros([total_index])

    dir_base = 'examples/optimal_frame_lying/'
    dir_files = os.listdir(dir_base+'data/')

    observation = env.reset()
    
    for i_ in range(42):
        if i_ in [2,5,12,18,25,28,35,40]:
            dir_file_name = dir_base + 'data/' + dir_files[i_]
            pkl_files = os.listdir(dir_file_name)
            for j_ in range(2, 50, 4):
                pkl_file_name = dir_file_name +'/'+ pkl_files[j_]
                evaluation_orient = [0.0,0.0,1.0]
                env.f_name = pkl_file_name
                env.set_file_name(pkl_file_name)
                #print(pkl_file_name)
                env.change_human_pose()
                #print('Evaluation point: ',evaluation_point)
                score_array[index] = optimizing_across_poses(point, env, frame_i)
                #print('score index ', score_array[index] )
                index=index+1
                #print('At index ', index)    
                time.sleep(1)

    f_value = np.sum(score_array)
    print('--Iteration done: f_value: ', f_value,' index ',index)
    return f_value        


def bound_offset(env, frame_i):
    
    human_base = np.array(env.human.get_pos_orient(env.human.base)[0],dtype="float64")

    head_point = np.array(env.human.get_pos_orient(env.human.head)[0],dtype="float64")
    neck_point = np.array(env.human.get_pos_orient(env.human.neck)[0],dtype="float64")
    chest_point = np.array(env.human.get_pos_orient(env.human.right_pecs)[0],dtype="float64")
    waist_point = np.array(env.human.get_pos_orient(env.human.waist)[0],dtype="float64")
    stomach_point = np.array(env.human.get_pos_orient(env.human.waist)[0],dtype="float64")
    left_shoulder_point = np.array(env.human.get_pos_orient(env.human.left_shoulder)[0],dtype="float64")
    right_shoulder_point = np.array(env.human.get_pos_orient(env.human.right_shoulder)[0],dtype="float64")
    left_elbow_point = np.array(env.human.get_pos_orient(env.human.left_elbow)[0],dtype="float64")
    right_elbow_point = np.array(env.human.get_pos_orient(env.human.right_elbow)[0],dtype="float64")

    target_points = np.zeros([9,3])

    target_points[0] = human_base - head_point 
    target_points[1] = human_base - neck_point 
    target_points[2] = human_base - chest_point
    target_points[3] = human_base - waist_point 
    target_points[4] = human_base - stomach_point 
    target_points[5] = human_base - left_shoulder_point 
    target_points[6] = human_base - right_shoulder_point 
    target_points[7] = human_base - left_elbow_point 
    target_points[8] = human_base - right_elbow_point 

    return target_points[frame_i]


def show_bounds(env, arm_length, bnd_point, frame_i, bnd_weight):

    point=np.array([0,0,0])
    head_point = point_wrt_body_frames(point, env, frame_i)

    x_min = bnd_point[0]+head_point[0] +bnd_weight[0]*arm_length
    x_max = bnd_point[0]+head_point[0] +bnd_weight[1]*arm_length

    y_min = bnd_point[1]+head_point[1] +bnd_weight[2]*arm_length
    y_max = bnd_point[1]+head_point[1] +bnd_weight[3]*arm_length

    z_min = bnd_point[2]+head_point[2] +bnd_weight[4]*arm_length
    z_max = bnd_point[2]+head_point[2] +bnd_weight[5]*arm_length

    x_ = np.linspace(x_min,x_max,8)
    y_ = np.linspace(y_min,y_max,4)
    z_ = np.linspace(z_min,z_max,8)

    X, Y, Z = np.meshgrid(x_ ,y_, z_) 

    x=X.flatten()
    y=Y.flatten()
    z=Z.flatten()

    points_sphere = np.array([x,y,z])
    points_sphere = points_sphere.transpose()

    env.create_spheres(radius=0.01, mass=0.0, batch_positions=points_sphere, visual=True, collision=False, rgba=[0, 1, 1, 1])

    print('Points------------------------')
    print(points_sphere.shape)


def get_wrist_wrt_base(env):
    wrist_point = np.array(env.human.get_pos_orient(env.human.right_wrist)[0],dtype="float64")
    human_base_point = np.array(env.human.get_pos_orient(env.human.base)[0],dtype="float64")
    return wrist_point-human_base_point


def cma_ws_optimizer(frame_i,env):

    arm_length = 0.525
    #x0 = np.array([-0.1, -0.1, 0 ])
    x0 = get_wrist_wrt_base(env)

    pop_size = 12

    bnd_point = bound_offset(env, frame_i)
    
    x_init = (x0[0]+bnd_point[0],x0[1]+bnd_point[1],x0[2]+bnd_point[2])
    #bnd should be same as in show_bounds
    bnd_weight = np.array([-0.8, -0.0, -0.6, 0.4, -0.5, +1.0])
    bnd = ((bnd_point[0]+bnd_weight[0]*arm_length,bnd_point[0]+bnd_weight[1]*arm_length),(bnd_point[1]+bnd_weight[2]*arm_length,bnd_point[1]+bnd_weight[3]*arm_length),(bnd_point[2]+bnd_weight[4]*arm_length,bnd_point[2]+bnd_weight[5]*arm_length))
     
    #show_bounds(env, arm_length, bnd_point, frame_i, bnd_weight)
    bounds = np.array([[bnd_point[0]+bnd_weight[0]*arm_length,bnd_point[0]+bnd_weight[1]*arm_length],[bnd_point[1]+bnd_weight[2]*arm_length,bnd_point[1]+bnd_weight[3]*arm_length],[bnd_point[2]+bnd_weight[4]*arm_length,bnd_point[2]+bnd_weight[5]*arm_length] ])
    #sigma_ = (bnd_point[0]+bnd_weight[1]*arm_length - bnd_point[1]+bnd_weight[2]*arm_length) / 5  #before 0.1
    sigma_ = 0.2
    optimizer = CMA(mean=np.array(x_init), sigma=sigma_, population_size=pop_size, bounds=bounds)
    fevals = 0
    iteration = 0
    t0 = time.time()
    
    best_point = x0
    best_cost = float('inf')

    while iteration < 2:
        iteration += 1
        solutions = []
        print('optimizer size',optimizer.population_size)
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            cost = optimizing_across_poses(x,env,frame_i)
            #cost = iterating_across_poses(x,env,frame_i)
            if cost < best_cost:
                best_cost = cost
                best_point = x

            t1 = time.time()
            elapsed_time = t1 - t0
            solutions.append((x, cost))
            fevals += 1
            print("iteration:", iteration, "fevals:", fevals, "elapsed_time:", elapsed_time, "reward:", cost, "action:", x)
            #pickle.dump({"iteration": iteration, "fevals": fevals, "elapsed_time": elapsed_time, "reward": -cost, "action": x}, f)
        optimizer.tell(solutions)
        
        mean = np.mean([cost for x, cost in solutions])
        min = np.min([cost for x, cost in solutions])
        max = np.max([cost for x, cost in solutions])
        print(f"Iteration: {iteration}, fevals: {fevals}, elapsed time: {elapsed_time:.2f}, mean reward = {mean:.2f}, min/max reward = {min:.2f}/{max:.2f}")

        if optimizer.should_stop():
            break

    print('For frame ', frame_i,' cost is ', best_cost, ' and solution is ', best_point)
    env.create_spheres(radius=0.06, mass=0.0, batch_positions=best_point, visual=True, collision=False, rgba=[1, 0, 0, 1])
    
    t_end = time.time()
    print("-----------------Total duration", t_end-t0)
    return best_cost, best_point


with open('file_modified_handover_40_second_cma2.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

print(loaded_dict)
points = np.array(loaded_dict['Points']) # 192*3   
score = np.array(loaded_dict[' score ']) # 192


# score  size (256,6)
print('Evaluation analysis:  (lower means better)')
print('score_wrt_head'  , score[0]) 
print('score_wrt_neck'  , score[1]) 
print('score_wrt_chest'  , score[2]) 
print('score_wrt_waist'  , score[3]) 
print('score_wrt_stomach'  , score[4]) 
print('score_wrt_left_shoulder'  , score[5]) 
print('score_wrt_right_shoulder'  , score[6]) 
print('score_wrt_left_elbow'  , score[7]) 
print('score_wrt_right_elbow'  , score[8]) 


parts = ['Head', 'Neck', 'Chest', 'Waist', 'Stomach', 'Left_Shoulder', 'Right_Shoulder', 'Left_elbow', 'Right_elbow' ]
func = [ score[0],  score[1],  score[2],  score[3],  score[4],  score[5],  score[6],  score[7],  score[8]]
# courses = list(data.keys())
# values = list(data.values())
if __name__ == "__main__":

    env_name = "HumanResting-v1"
    env = make_env(env_name, coop=True)
    coop=True
    #env = gym.make()
    env.render()
    env.setup_camera()

    observation = env.reset()

    human_base_pose, human_orient = env.human.get_pos_orient(env.human.base) 
    orientation = p.getEulerFromQuaternion(np.array(human_orient), physicsClientId=env.id) 
                
    score = np.zeros([9])
    point_optimized = np.zeros([9,3])

    frame_i = 3
    point_optimized = points[frame_i][:3] - np.array([-0.2,-0.20,-0.15])
    orient_ = [0,0,0]
    
    #score[frame_i] = run_the_simulation(point_optimized, frame_i, env, target, orient_)
    
    cost, point_optimized = cma_ws_optimizer(frame_i, env)
    dict_item = { ' score ' : cost  }
    evaluation_point = point_wrt_body_frames(point_optimized, env, frame_i)
    target = env.create_sphere(radius=0.06, mass=0.0, pos=point_optimized, visual=True, collision=False, rgba=[1, 0, 0, 1]) #human hand
    target.set_base_pos_orient(evaluation_point, orient_)
    #c_ = run_the_simulation(point_optimized, frame_i, env, target, orient_)
    c_ = optimizing_across_poses(point_optimized,env,frame_i)
    f = open("result1_handover.pkl","wb")
    
    #print('Evaluation point: ',evaluation_point)
    
    #f = open("result1_showscreen.pkl","wb")
    #pickle.dump(dict_item,f)
    
    #mid_angle = np.array([-1.28484584e-14,  1.74532925e-01,  1.24455499e-14, -8.98027383e-02, -1.10828408e+00,  
    #          6.82478992e-14, -1.11701072e+00, -1.43938821e-14,  7.85398163e-02,  1.74532925e-01])

    #env.human.set_joint_angles(env.human.right_arm_joints, mid_angle)
    p.stepSimulation(physicsClientId=env.id)

    time.sleep(6)
    f.close()

# 1min 15 sec with render

    # f2 [-0.02636151 -0.20183839  0.26625905]  chest
    # f3 [-0.02562699  0.13809083  0.26250002]  waist


# solution is [-0.1364918  -0.00968062  0.30648776] frame_i=3 try1 with render
# solution is [-0.22079414  0.36095911  0.49786121] frame_i=3 try2 with render time_difference:73.70984101295471
# solution is [-0.39491915  0.06364492  0.27920154] frame_i=3 try3 without render time_difference: 66.1298747062683
# solution is [-0.33313673 -0.04773659  0.2941261 ] frame_i=3 try2 with render time_difference: 93.70984101295471
# screenshot online opt-----------------Total duration 81.9611246585846 /40 sec



# if f_moment=0 For frame  3  cost is  25.157534667564562  and solution is  [-0.11035453  0.03483751  0.40693261]
# if f_moment=/0 For frame  3  cost is  30.982856475512108  and solution is  [-0.04375808  0.41050703  0.49725392]
#   mid_angle = np.array([-1.28484584e-14,  1.74532925e-01,  1.24455499e-14, -8.98027383e-02, -1.10828408e+00,  
#              6.82478992e-14, -1.11701072e+00, -1.43938821e-14,  7.85398163e-02,  1.74532925e-01])
# env.human.set_joint_angles(env.human.right_arm_joints, mid_angle)