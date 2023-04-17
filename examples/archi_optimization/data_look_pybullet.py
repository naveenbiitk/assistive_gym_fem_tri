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
import time

env_name = "ObjectHandoverStretchHuman-v1"
env = make_env(env_name, coop=True)
# env = gym.make()
env.render()
observation = env.reset()

base_pos_set = [5,-5,0.1]
quat_orient = env.get_quaternion([0.0, 0.0, 3.14])
env.robot.set_base_pos_orient( base_pos_set, quat_orient)

p.removeBody(1 , physicsClientId=env.id)


with open('file_handover_2.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

sphere_points = np.array(loaded_dict['Points']) #192*3   
det_list = np.array(loaded_dict['determinant'])  #192      d
#jlwki_list = np.array(loaded_dict['jlwki'])      #192      d
#forces_list = np.array(loaded_dict['forcemom']) #192*10*6  
sforce_list = np.array(loaded_dict['sum_force']) #192*10
smoment_list = np.array(loaded_dict['sum_moment']) #192*10
check_mask_list = np.array(loaded_dict['reach_check'])
f_moment = np.array(loaded_dict['fobj_moment'])
f_angle_list = np.array(loaded_dict['fobj_angle'])

shoulder_pos,shoulder_orient = env.human.get_pos_orient(env.human.right_shoulder)
print('Shoulder pos', shoulder_pos)

#print('sphere_points',np.max(sphere_points[i]))

print('loop started')
# for i_ in range(n):
# 	pos = [sphere_points[i_]]
# 	if check_mask_list[i_]==1:
# 		env.create_sphere(radius=0.01, mass=0.0, pos=pos, visual=True, collision=False, rgba=[0, 1, 1, 1])
# 	else:
# 		env.create_sphere(radius=0.01, mass=0.0, pos=pos, visual=True, collision=False, rgba=[1, 0, 1, 1])


sphere_points_new = sphere_points[check_mask_list==1][:]

f_moment_new = f_moment[check_mask_list==1]
f_angle_list_new = f_angle_list[check_mask_list==1]
det_list_new = det_list[check_mask_list==1]

print('Max mom',np.max(f_moment_new))
print('Max ang',np.max(f_angle_list_new))
print('Max det',np.max(det_list_new))

for i_ in range(len(sphere_points_new)):
	print('Iteration', i_)
	pos_i = sphere_points_new[i_]
	alpha_1 = f_moment_new[i_]/np.max(f_moment_new)
	alpha_2 = f_angle_list_new[i_]/np.max(f_angle_list_new)
	beta_3 = det_list_new[i_]/np.max(det_list_new)
	alpha_3 = 1-beta_3
	alpha = (alpha_1+alpha_2+0.5*alpha_3)/2.5
	env.create_sphere(radius=0.01, mass=0.0, pos=pos_i, visual=True, collision=False, rgba=[alpha, 1-alpha, 1, 1])
	p.stepSimulation(physicsClientId=env.id)


print('Loop ended')
# pos2 = sphere_points[check_mask_list==1][:]
# env.create_spheres(radius=0.01, mass=0.0, batch_positions=pos2, visual=True, collision=False, rgba=[0, 1, 1, 1])


for _ in range(50):
    p.stepSimulation(physicsClientId=env.id)



time.sleep(30)
print('Done')