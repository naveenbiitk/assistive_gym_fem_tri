import gym, assistive_gym, argparse
import pybullet as p
import numpy as np

import os
import ray._private.utils
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter

from assistive_gym.learn_human import make_env

env_name = "HumanTesting-v1"
env = make_env(env_name, coop=True)
#env = gym.make()
env.render()
env.seed(0)
observation = env.reset()


for eps_id in range(10):
    obs = env.reset()
    # print("OBS: ", obs)
    # obs= np.concatenate((obs['robot'], obs['human']))
    # print("OBS: ", obs)

    prev_action = np.zeros_like(env.action_space.sample())
    prev_reward = 0
    #done = {'robot':False, 'human':False}
    done = False
    t = 0
    # while (not done) and (t < 150):

        
    
    while not done and (t < 50):
        target_pose_angles_hand = env.human.ik(env.human.right_wrist, env.target_pos, None, ik_indices=env.human.controllable_joint_indices, max_iterations=4000)

        env.render()
        human_action = np.zeros(6)
        final_action =   human_action*80

        # observation, reward, done, info = env.step(action*100)
        #print('final_action', final_action)
        new_obs, rew, done, info = env.step(final_action)

        k1 = 0.08

        #print('set angles', set_target_angles)
        curr = np.array(env.human.get_joint_angles(env.human.controllable_joint_indices) )
        error = (target_pose_angles_hand-curr)*k1
        move_to = curr + error
        env.human.set_joint_angles(env.human.controllable_joint_indices, move_to)

        obs = new_obs
        prev_action = final_action
        prev_reward = rew
        t += 1
        #print(t)

    # writer.write(batch_builder.build_and_reset())
