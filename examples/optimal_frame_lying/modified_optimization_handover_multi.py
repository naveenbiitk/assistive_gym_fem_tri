import gym, assistive_gym, argparse, sys, multiprocessing, time
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

from scipy import optimize

import cma
from cma.optimization_tools import EvalParallel2

import tensorflow as tf
import logging
import os

index = 0

#frame_i = 0




def run_simulation_pics(env_human, env_id, count_target):
    global index
    coop = True
    count_=0
    t=0
    while count_<count_target:
        
        #env.render()
        p.removeAllUserDebugItems()
        head_t_pos, head_t_orient = env_human.get_pos_orient(env_human.head)
        eye_pos = [0, -0.11, 0.1] if env_human.gender == 'male' else [0, -0.1, 0.1] # orientation needed is [0, 0, 0.707, -0.707]
        heye_pos, heye_orient = p.multiplyTransforms(head_t_pos, head_t_orient, eye_pos, [0, 0, 0, 1], physicsClientId=env.id)
        #generate_line(heye_pos, heye_orient)
         # t = p.addUserDebugText(text=st, textPosition=[-0.8,0,1], textColorRGB=[1,1,1], textSize=2, lifeTime=0.5, physicsClientId=env.id )
        human_action = np.zeros(env.action_human_len)
        
        observation, reward, done, info = env.step(human_action)
        count_ = count_ + 1

        if count_==5:
           img, depth = env.get_camera_image_depth()
           im = Image.fromarray(img)
           st = str(index)
           im.save('human_sample/'+st+'.png')



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

    observation = env.reset()

    human_base_pose, human_orient = env.human.get_pos_orient(env.human.base) 
    orientation = p.getEulerFromQuaternion(np.array(human_orient), physicsClientId=env.id) 

    index = 0
    total_index=320
    score_array = np.zeros([total_index])

    for h_i_ in range(-3,-1,2):   #head_x
        for h_j_ in range(-3,-1,2):  #head_z
            for j_ in range(-3,4,2):   # waist orientation
                for i_ in range(-3,4,2):  # base orientation
                    i = i_*10
                    j = j_*10
                    h_i = h_i_*10
                    h_j = h_j_*10
                    orientation_new = [orientation[0], orientation[1]+(i/180*3.14), orientation[2]+(i/180*3.14)]
                    new_human_orient = p.getQuaternionFromEuler(orientation_new, physicsClientId=env.id)
                    env.human.set_base_pos_orient(human_base_pose, new_human_orient )
                    joints_positions = [ (env.human.j_waist_z, j)]# -30 to +30
                    env.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None)
                    #print('scipy point: ',point)
                    evaluation_point = point_wrt_body_frames(point, env, frame_i)
                    #print('Evaluation point: ',evaluation_point)
                    score_array[index] = cmaes_handover.optimizing_function(evaluation_point, env.human, env.id, env) #evaluate_object_handover(evaluation_point)
                    index=index+1
                    #print('At index ', index)

    f_value = np.sum(score_array)
    print('Iteration done: f_value: ', f_value)
    return f_value                



# def generate_bounds(frame_i, env):
def show_bounds(env, arm_length, bnd_point, frame_i):

    point=np.array([0,0,0])
    head_point = point_wrt_body_frames(point, env, frame_i)

    x_min = bnd_point[0]+head_point[0] - 1.5*arm_length 
    x_max = bnd_point[0]+head_point[0] + 0.5*arm_length

    y_min = bnd_point[1]+head_point[1] - 2.0*arm_length
    y_max = bnd_point[1]+head_point[1] - 1.0*arm_length

    z_min = bnd_point[2]+head_point[2] - arm_length
    z_max = bnd_point[2]+head_point[2] + arm_length

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



def bound_offset(env, frame_i):
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

    target_points[0] = right_shoulder_point - head_point 
    target_points[1] = right_shoulder_point - neck_point 
    target_points[2] = right_shoulder_point - chest_point
    target_points[3] = right_shoulder_point - waist_point 
    target_points[4] = right_shoulder_point - stomach_point 
    target_points[5] = right_shoulder_point - left_shoulder_point 
    target_points[6] = right_shoulder_point - right_shoulder_point 
    target_points[7] = right_shoulder_point - left_elbow_point 
    target_points[8] = right_shoulder_point - right_elbow_point 

    return target_points[frame_i]



def scipy_optimizer(frame_i, env, return_dict_1, return_dict_2):
    
    with open('file_modified_handover_40.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)

    arm_length = 0.525-0.120

    #x0 = np.array([-0.1, -0.1, 0 ])
    points_old = np.array(loaded_dict['Points'])
    x0 = points_old[frame_i] - np.array([0,0,0.15])
    bnd_point = bound_offset(env, frame_i)

    bnd = ((bnd_point[0]-1.5*arm_length,bnd_point[0]+0.5*arm_length),(bnd_point[1]- 1.5*arm_length,bnd_point[1]- 0.5*arm_length),(bnd_point[2]+0.7*arm_length,bnd_point[2]+1.2*arm_length))

    #bnd = ((bnd_point[0]-1.5*arm_length,bnd_point[0]+0.5*arm_length),(bnd_point[1]- 2.0*arm_length,bnd_point[1]- 1.0*arm_length),(bnd_point[2]-arm_length,bnd_point[2]+arm_length))

    #show_bounds(env, arm_length, bnd_point, frame_i)
    #time.sleep(1)
    print('Initial point: ',x0)
    print('Bounds ',bnd)
    opt = {'disp':True,'maxiter':25} 
    result = optimize.minimize(fun = optimizing_across_poses, x0=x0, args=(env, frame_i), method='Nelder-Mead', bounds=bnd ,options=opt )
    print('For frame ', frame_i,' cost is ', result.fun, ' and solution is ', result.x)
    print('---------------------------------------------------------------------------') 
    
    return_dict_1[frame_i]=result.fun
    return_dict_2[frame_i]=result.x

    return 



def logging_function(cma, logger):
    fitness = cma.best_fitness()

    # Write best fitness to the tensorboard summary log
    with summary_writer.as_default():
        tf.summary.scalar('fitness', fitness, step=cma.generation)

    # Periodically log progress
    if cma.generation % 10 == 0:
        logger.info(f'Generation {cma.generation} - fitness {fitness}')

    if cma.termination_criterion_met or cma.generation == max_epochs:
        sol = cma.best_solution()
        logger.info(f'Final solution at gen {cma.generation}: {sol} (fitness: {fitness})')


def cma_optimizer(frame_i, env):

    with open('file_modified_handover_40.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)

    arm_length = 0.525-0.120

    points_old = np.array(loaded_dict['Points'])
    x0 = points_old[frame_i] - np.array([0,0,0.15])


    num_proc = multiprocessing.cpu_count()#16

    bnd_point = bound_offset(env, frame_i)

    bnd = ((bnd_point[0]-1.5*arm_length,bnd_point[0]+0.5*arm_length),(bnd_point[1]- 1.5*arm_length,bnd_point[1]- 0.5*arm_length),(bnd_point[2]+0.7*arm_length,bnd_point[2]+1.2*arm_length))

    opts = cma.CMAOptions({'verb_disp': 1, 'popsize': num_proc})
    opts.set('bounds',bnd)

    opts.set('iteration', 1)
    opts['iteration'] = 1

    cma_option = {"BoundaryHandler": cma.BoundTransform,
                "bounds": bnd,
            }

    opts.update(cma_option)
    sigma0=0.2

    t0 = time.time()
    es = cma.CMAEvolutionStrategy(x0,sigma0 )

    iterations = 0
    fevals=0

    with EvalParallel2(optimizing_across_poses, number_of_processes=num_proc) as eval_all:
        while not es.stop():
            iterations += 1
            fevals += num_proc

            point_lists = es.ask()
            output = eval_all(point_lists)
            t1 = time.time()

            output = [list(x) for x in zip(*output)]
            costs = output[0]
            observations = output[1]
            elapsed_time = output[2]
            es.tell(point_lists, costs)
            rewards = [-c for c in costs]
            mean_rewards = np.mean(rewards)
            print(f"Iteration: {iterations}, fevals: {fevals}, elapsed time: {total_elapsed_time:.2f}, mean reward = {mean_reward:.2f}")
            
            if np.any(np.array(costs) <= -reward_threshold): break


    es.result_pretty()
    f_value = es.result()[1]
    solution = es.result()[0]
    print('best f-value =',  f_value)
    print('best solution =', solution)
    print('---------------------------------------------------------------------------') 
    return f_value, solution


#run cmeas to find the optimal point

if __name__ == "__main__":

    manager = multiprocessing.Manager()
    return_dict_1 = manager.dict()
    return_dict_2 = manager.dict()

    env_name = "HumanLying-v1"
    env = make_env(env_name, coop=True)
    coop=True

    #env = gym.make()
    #env.render()
    env.setup_camera()
    observation = env.reset()

    #observation= np.concatenate((observation['robot'], observation['human']))
    #env.robot.print_joint_info()    

    human_base_pose, human_orient = env.human.get_pos_orient(env.human.base) 
    orientation = p.getEulerFromQuaternion(np.array(human_orient), physicsClientId=env.id) 
    #print('orientation yaw',orientation[2])
                
    score = np.zeros([9])
    point_optimized = np.zeros([9,3])
    jobs = []
    # frame_i=8
    # scipy_optimizer(frame_i, env)
    for frame_j in range(9):
        frame_i = frame_j

        pr = multiprocessing.Process(target=scipy_optimizer, args=(frame_j, env, return_dict_1, return_dict_2))
        jobs.append(pr)
        pr.start()

    for proc in jobs:
        proc.join()

    dict_item = { 'Points': return_dict_1,' score ' : return_dict_2  }
    f = open("file_modified_handover_40_second_scipy_multi.pkl","wb")
    pickle.dump(dict_item,f)
    f.close()


