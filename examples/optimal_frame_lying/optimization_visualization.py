import matplotlib.pyplot as plt
import pickle
import gym, assistive_gym, argparse
import pybullet as p
import numpy as np
import yaml

from assistive_gym.learn import make_env
from math import sin,cos
from PIL import Image
import math
import time
import pickle

import cmaes_handover 
import cmaes_showphone




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
  f_moment = f_moment #/(max_moment**2)

  f_angle = np.sum( (target_pose_angles-mid_angle)**2 )
  f_angle = f_angle/(max_ang**2)

  current_position = env.human.get_pos_orient(env.human.right_wrist)[0]
  f_position = (target_position-current_position)**2

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

    if best_sol is None or f_sol < best_sol:
      best_sol = f_sol
      best_joint_positions = target_joint_angles_h

    return best_joint_positions






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



def run_the_simulation(point_optimized, frame_i, env, target, orient_):
    evaluation_point = point_wrt_body_frames(point_optimized, env, frame_i)
    #print('Evaluation point: ',evaluation_point)
    target.set_base_pos_orient(evaluation_point, [0, 0, 0, 1])
    #score_array = cmaes_showphone.optimizing_function(evaluation_point, orient_, env.human, env.id, env) #evaluate_object_handover(evaluation_point)
    score_array = cmaes_handover.optimizing_function(evaluation_point, env.human, env.id, env) #evaluate_object_handover(evaluation_point)    
    print('score_array',score_array)
    return score_array 



with open('file_modified_handover_40_second_cma2.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

# with open('file_modified_showscreen_eye.pkl', 'rb') as f:
#     loaded_dict = pickle.load(f)

#{ 'Points': point_optimized,' score ' : score  }

print(loaded_dict)
points = np.array(loaded_dict['Points']) #192*3   
score = np.array(loaded_dict[' score '])  #192      d


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

    env_name = "HumanLying-v1"
    env = make_env(env_name, coop=True)
    coop=True
    #env = gym.make()
    env.render()
    env.setup_camera()

    observation = env.reset()
    #env.human.set_gravity(0, 0, 0)
    #observation= np.concatenate((observation['robot'], observation['human']))
    #env.robot.print_joint_info()

    human_base_pose, human_orient = env.human.get_pos_orient(env.human.base) 
    orientation = p.getEulerFromQuaternion(np.array(human_orient), physicsClientId=env.id) 
    #print('orientation yaw',orientation[2])
                
    score = np.zeros([9])
    point_optimized = np.zeros([9,3])

    target = env.create_sphere(radius=0.02, mass=0.0, pos=[0,0,0], visual=True, collision=False, rgba=[1, 1, 0, 1]) #human hand

    for frame_i in range(9):
        if frame_i!=11: # or frame_i==7:
            point_optimized = points[frame_i][:3] - np.array([-0.2,-0.20,-0.15])
            #orient_ = points[frame_i][3:6]
            orient_ = [0,0,0]
            score[frame_i] = run_the_simulation(point_optimized, frame_i, env, target, orient_)
            #run_simulation(env, point_optimized, frame_i, env, target)
        
        #score[frame_i], point_optimized[frame_i] = scipy_optimizer(frame_i, env)

    dict_item = { ' score ' : score  }
    f = open("result1_handover.pkl","wb")
    #f = open("result1_showscreen.pkl","wb")
    #pickle.dump(dict_item,f)
    f.close()


