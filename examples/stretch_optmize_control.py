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


env_name = "ObjectHandoverStretch-v1"
env = make_env(env_name, coop=True)
#env = gym.make()
env.render()
observation = env.reset()
#observation= np.concatenate((obs))


####################################################################
####################################################################
joint_indices = env.robot.controllable_joint_indices
print('controllable_joints',joint_indices)
lower_limits = np.array([env.robot.lower_limits[ind] for ind in joint_indices])
upper_limits = np.array([env.robot.upper_limits[ind] for ind in joint_indices])
joint_ranges = upper_limits - lower_limits
joint_centers = lower_limits + (joint_ranges / 2.)

print('upper_limits', upper_limits)
print('lower_limits', lower_limits)
print('joint_centers', joint_centers)

count_=0
while count_<50:
    env.render()
 
    # current_joint_angles = env.human.get_joint_angles(env.human.right_arm_joints)
    # env.human.set_joint_angles(env.human.right_arm_joints, target_pose_angles*0)
    # env.human.set_joint_angles(env.human.right_arm_joints, right_arm_angles_end)

    #kp=0.5
    #if count_>0:
    #    err_angles = target_pose_angles-current_joint_angles
    #else:
    #    err_angles = right_arm_angles_end - current_joint_angles
    
    #action = kp*(err_angles)
    
    # print('current joint angles',current_joint_angles)

    # current_joint_angles = env.robot.get_joint_angles(env.robot.left_gripper_indices)
    # action[len(env.robot.wheel_joint_indices):] = target_pose_angles-current_joint_angles

    env.robot.set_joint_angles(joint_indices,upper_limits)
    action_ = env.action_space_robot.sample()*0 # Get a random action

    observation, reward, done, info = env.step(action_)
    # reward_total += reward
    # task_success = info['task_success']
    count_ = count_+1

#print('reward')
#print(reward_total)
#print('Current joint angles', current_joint_angles )
print('task_success')
#print(task_success)


