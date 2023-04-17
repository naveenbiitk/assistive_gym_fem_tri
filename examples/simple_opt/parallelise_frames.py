import multiprocessing
import single_optimization
import gym, assistive_gym, argparse
import pybullet as p
import numpy as np
import yaml
import pickle
import time

from assistive_gym.learn import make_env
# Main function to be parallelized
with open('file_modified_handover_40_second_cma2.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

print(loaded_dict)
points = np.array(loaded_dict['Points']) # 192*3   

def main(frame_i):
    start = time.time()
    print("Running process", frame_i)
    print('-----This is frame ', frame_i, '----------------')
    env_name = "HumanResting-v1"
    env = make_env(env_name, coop=True)
    coop=True
    env.setup_camera()
    observation = env.reset()
    human_base_pose, human_orient = env.human.get_pos_orient(env.human.base) 
    orientation = p.getEulerFromQuaternion(np.array(human_orient), physicsClientId=env.id) 
    score,point_optimized = single_optimization.cma_ws_optimizer(frame_i, env)
    #cost, point_optimized = cma_ws_optimizer(frame_i, env)
    dict_item = { 'Points': point_optimized,' score ' : score, 'frame_i': frame_i  }
    file_base_name = "cane_frame_analysis_jan_24_seq"+str(frame_i)+".pkl"
    f = open(file_base_name,"wb")
    pickle.dump(dict_item,f)
    f.close()
    end = time.time()
    print("Time taken for frame ", frame_i, " is ", end-start)
# Create a list of arguments to pass to the main function
arg_list = [i for i in range(9)]

# Create a pool of 9 worker processes
with multiprocessing.Pool(9) as p:
    # Map the main function to the list of arguments
    p.map(main, arg_list)
