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
#import cmaes_showphone

with open('file_modified_handover_40_second_cma2.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

#with open('file_modified_showscreen_eye.pkl', 'rb') as f:
#    loaded_dict = pickle.load(f)

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


parts = ['Head', 'Neck', 'Chest', 'Waist', 'Stomach', 'Left_Should', 'Right_Should', 'Left_elbow', 'Right_elbow' ]
func = [ score[0],  score[1],  score[2],  score[3],  score[4],  score[5],  score[6],  score[7],  score[8]]
# courses = list(data.keys())
# values = list(data.values())


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


def print_target_name(frame_i):
    if frame_i==0:
        print('--head_point = np')
    if frame_i==1:
        print('--neck_point = np')
    if frame_i==2:
        print('--chest_point = n')
    if frame_i==3:
        print('--waist_point = n')
    if frame_i==4:
        print('--stomach_point =')
    if frame_i==5:
        print('--left_shoulder_p')
    if frame_i==6:
        print('--right_shoulder_')
    if frame_i==7:
        print('--left_elbow_poin')
    if frame_i==8:
        print('--right_elbow_poi')


def run_the_simulation(point_optimized, frame_i, env, target):
    evaluation_point = point_wrt_body_frames(point_optimized, env, frame_i)
    #print('Evaluation point: ',evaluation_point)
    target.set_base_pos_orient(evaluation_point, [0, 0, 0, 1])
    #score_array = cmaes_showphone.optimizing_function(evaluation_point, env.human, env.id, env) #evaluate_object_handover(evaluation_point)
    score_array = cmaes_handover.optimizing_function(evaluation_point, env.human, env.id, env) #evaluate_object_handover(evaluation_point)    
    print('score_array',score_array) 




if __name__ == "__main__":

    env_name = "HumanTesting-v1"
    env = make_env(env_name, coop=True)
    coop=True
    #env = gym.make()
    env.render()
    env.setup_camera()
    observation = env.reset()
    env.human.set_gravity(0, 0, -10)
    #observation= np.concatenate((observation['robot'], observation['human']))
    #env.robot.print_joint_info()

    human_base_pose, human_orient = env.human.get_pos_orient(env.human.base) 
    orientation = p.getEulerFromQuaternion(np.array(human_orient), physicsClientId=env.id) 
    #print('orientation yaw',orientation[2])
    env.convert_smpl_body_to_gym()
    score = np.zeros([9])
    point_optimized = np.zeros([9,3])

    target = env.create_sphere(radius=0.02, mass=0.0, pos=[0,0,0], visual=True, collision=False, rgba=[1, 1, 0, 1]) #human hand

    for frame_i in range(9):
        if frame_i!=11: # or frame_i==7:
            if frame_i==3:
                point_optimized = points[frame_i]+ np.array([0,0,0.1])
            else:
                point_optimized = points[frame_i]    
            
            print_target_name(frame_i)
            run_the_simulation(point_optimized, frame_i, env, target)
        #score[frame_i], point_optimized[frame_i] = scipy_optimizer(frame_i, env)

    # dict_item = { 'Points': point_optimized,' score ' : score  }
    # f = open("file_modified_showscreen_eye.pkl","wb")
    # pickle.dump(dict_item,f)
    # f.close()


