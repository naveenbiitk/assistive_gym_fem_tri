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

from scipy import optimize
import cma

index = 0

def calculate_obj_ik(env, target_pose_angles, target_position):

  mid_angle = np.array([-1.28484584e-14,  1.74532925e-01,  1.24455499e-14, -8.98027383e-02, -1.10828408e+00,  
              6.82478992e-14, -1.11701072e+00, -1.43938821e-14,  7.85398163e-02,  1.74532925e-01])
  
  max_moment = 363.4084744
  max_ang = 16.7190253038

  weights_opt_ik = np.array([ 0.5,  0.5,  0.1 ])

  forces = np.array([env.human.get_force_torque_sensor(joints_j) for joints_j in env.human.right_arm_joints])

  linear_forces = np.sqrt( forces[:,0]*forces[:,0] + forces[:,1]*forces[:,1] + forces[:,2]*forces[:,2] )
  linear_moment = np.sqrt( forces[:,3]*forces[:,3] + forces[:,4]*forces[:,4] + forces[:,5]*forces[:,5] )
  
  #f_moment = np.sum(linear_moment**2)
  #f_moment = f_moment/(max_moment**2)

  f_moment = np.sum(forces[:7,5]**2)/5
  f_moment = f_moment  #/(max_moment)

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


def run_simulation_pics(env_human, env_id, count_target):
    global index
    coop = True
    count_=0
    t=0
    while count_<count_target:
        
        # env.render()
        p.removeAllUserDebugItems()
        head_t_pos, head_t_orient = env_human.get_pos_orient(env_human.head)
        eye_pos = [0, -0.11, 0.1] if env_human.gender == 'male' else [0, -0.1, 0.1] # orientation needed is [0, 0, 0.707, -0.707]
        heye_pos, heye_orient = p.multiplyTransforms(head_t_pos, head_t_orient, eye_pos, [0, 0, 0, 1], physicsClientId=env.id)
        # generate_line(heye_pos, heye_orient)
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
    total_index=256
    score_array = np.zeros([total_index])
    collision_objects = [env.furniture, env.human]
    collision_list = []
    for h_i_ in range(-3,4,2):   #head_x
        for h_j_ in range(-3,4,2):  #head_z
            for j_ in range(-3,4,2):   # waist orientation
                for i_ in range(-3,4,2):  # base orientation
                    i = i_*10
                    j = j_*10
                    h_i = h_i_*10
                    h_j = h_j_*10
                    orientation_new = [orientation[0], orientation[1], orientation[2]+(i/180*3.14)]
                    new_human_orient = p.getQuaternionFromEuler(orientation_new, physicsClientId=env.id)
                    env.human.set_base_pos_orient(human_base_pose, new_human_orient )
                    joints_positions = [ (env.human.j_waist_z, j), (env.human.j_head_x, h_i), (env.human.j_head_z, h_j)]# -30 to +30
                    env.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None)
                    #print('scipy point: ',point)
                    evaluation_point = point_wrt_body_frames(point, env, frame_i)
                    transform_pos = [evaluation_point[0]+0.1,evaluation_point[1],evaluation_point[2]+0.5]
                    tool_pos = evaluation_point #[point,point,point]
                    transform_orient = [0,0,0,1]
                    env.tool.set_base_pos_orient(tool_pos, transform_orient)
                    env.create_sphere(radius=0.02, mass=0.0, pos = transform_pos, visual=True, collision=False, rgba=[1, 1, 0, 1])
                    #print('Evaluation point: ',evaluation_point)
                    #score_array[index] = cmaes_handover.optimizing_function(transform_pos, env.human, env.id, env) #evaluate_object_handover(evaluation_point)
                    
                    dists_list = []
                    for obj in collision_objects:
                        dists = env.tool.get_closest_points(obj, distance=0)[-1]
                        dists_list.append(dists)

                    collision_detected = False
                    collision_score = 0
                    if all(not d for d in dists_list):
                        collision_detected = True
                        collision_score = 100
                    #   #print('----------Collision detected-------')

                    target_joint_angles_h, score = position_human_arm(env, transform_pos, attempts=20)
                    env.human.set_joint_angles(env.human.right_arm_joints, target_joint_angles_h)
                
                    #print('count_', count_, 'score', score, 'collision_flag', collision_flag)
                    score_array[index] = score+collision_score
                    #score_list.append(score)
                    collision_list.append(collision_detected)
                    #tool_pos_list.append(tool_pos)

                    index=index+1
                    #print('At index ', index)

    f_value = np.sum(score_array)
    print('Iteration done: f_value: ', f_value)
    return f_value                



# def generate_bounds(frame_i, env):
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



def bound_offset(env, frame_i):
    
    human_base = np.array(env.human.get_pos_orient(env.human.base)[0],dtype="float64")

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


def get_wrist_wrt_base(env):
    wrist_point = np.array(env.human.get_pos_orient(env.human.right_wrist)[0],dtype="float64")
    human_base_point = np.array(env.human.get_pos_orient(env.human.base)[0],dtype="float64")
    return wrist_point-human_base_point

def scipy_optimizer(frame_i, env):
    
    arm_length = 0.525
    #x0 = np.array([-0.1, -0.1, 0 ])
    x0 = get_wrist_wrt_base(env)
    #points_old = np.array(loaded_dict['Points'])
    #x0 = points_old[frame_i] - np.array([0,0,0])
    bnd_point = bound_offset(env, frame_i)

    x_init = (x0[0]+bnd_point[0],x0[1]+bnd_point[1],x0[2]+bnd_point[2])
    #bnd should be same as in show_bounds
    bnd_weight = np.array([-1.2, +0.5, -1.2, 0.0, -0.5, +0.5])
    bnd = ((bnd_point[0]+bnd_weight[0]*arm_length,bnd_point[0]+bnd_weight[1]*arm_length),(bnd_point[1]+bnd_weight[2]*arm_length,bnd_point[1]+bnd_weight[3]*arm_length),(bnd_point[2]+bnd_weight[4]*arm_length,bnd_point[2]+bnd_weight[5]*arm_length))
    #show_bounds(env, arm_length, bnd_point, frame_i, bnd_weight)
    #time.sleep(3)
    #return 
    print('Initial point: ',x_init)
    print('Bounds ',bnd)
    opt = {'disp':True,'maxiter':30} 
    result = optimize.minimize(fun = optimizing_across_poses, x0=x_init, args=(env, frame_i), method='Nelder-Mead', bounds=bnd ,options=opt )
    print('For frame ', frame_i,' cost is ', result.fun, ' and solution is ', result.x)
    print('---------------------------------------------------------------------------') 
    return result.fun, result.x


def cma_optimizer(frame_i, env):

    arm_length = 0.525-0.120

    x0 = np.array([-0.1, -0.1, 0 ])
    bnd_point = bound_offset(env, frame_i)

    bnd_weight = np.array([-1.2, +0.5, -1.2, 0.0, -0.5, +0.5])
    bnd = ((bnd_point[0]+bnd_weight[0]*arm_length,bnd_point[0]+bnd_weight[1]*arm_length),(bnd_point[1]+bnd_weight[2]*arm_length,bnd_point[1]+bnd_weight[3]*arm_length),(bnd_point[2]+bnd_weight[4]*arm_length,bnd_point[2]+bnd_weight[5]*arm_length))

    num_proc = multiprocessing.cpu_count()#16
    opts = cma.CMAOptions({'verb_disp': 1, 'popsize': num_proc})
    opts.set('bounds',bnd)

    opts.set('iteration', 1)
    opts['iteration'] = 1

    cma_option = {"BoundaryHandler": cma.BoundTransform,
                "bounds": bnd,
            }

    opts.update(cma_option)

    es = cma.CMAEvolutionStrategy(x0, 0.2)
    logger = cma.CMADataLogger().register(es)
    #es.optimize(optimizing_function, args=(env_human, test_set))
    es.optimize(optimizing_across_poses,iterations=10,args=(env, frame_i), opts=opts, callback=es.logger.plot)
    es.result_pretty()
    f_value = es.result()[1]
    solution = es.result()[0]
    print('best f-value =',  f_value)
    print('best solution =', solution)
    print('---------------------------------------------------------------------------') 
    return result[1], result[0]

#run cmeas to find the optimal point
# show_bounds

def show_bounds_wrt_torso(env, arm_length=0.5):
    torso_point = np.array(env.human.get_pos_orient(env.human.base)[0],dtype="float64")
    bnd_weight = np.array([-1.2, +0.5, -1.2, 0.0, -0.5, +0.5])

    x_min = torso_point[0] +bnd_weight[0]*arm_length
    x_max = torso_point[0] +bnd_weight[1]*arm_length

    y_min = torso_point[1] +bnd_weight[2]*arm_length
    y_max = torso_point[1] +bnd_weight[3]*arm_length

    z_min = torso_point[2] +bnd_weight[4]*arm_length
    z_max = torso_point[2] +bnd_weight[5]*arm_length

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

    print('Torso Points------------------------')
    print(points_sphere.shape)





if __name__ == "__main__":

    env_name = "HumanSitcane-v1"
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
                
    #show_bounds_wrt_torso(env, arm_length=0.5)

    #time.sleep(3)
    
    frame_i=8
    print('-----This is frame ', frame_i, '----------------')
    score,point_optimized = scipy_optimizer(frame_i, env)
    dict_item = { 'Points': point_optimized,' score ' : score, 'frame_i': frame_i  }
    file_base_name = "cane_frame_analysis_dec19_seq"+str(frame_i)+".pkl"
    f = open(file_base_name,"wb")
    pickle.dump(dict_item,f)
    f.close()

    score = np.zeros([9])
    point_optimized = np.zeros([9,3])

    # for frame_i in range(9):
    #    score[frame_i], point_optimized[frame_i] = scipy_optimizer(frame_i, env) #cma_optimizer(frame_i, env)

    # dict_item = { 'Points': point_optimized,' score ' : score  }
    # f = open("cane_frame_analysis_dec19.pkl","wb")
    # pickle.dump(dict_item,f)
    # f.close()


# Use the minimize function to find the minimum of the loss function
# res = minimize(loss, x0=0, callback=lambda x: print(loss(x)))
# Print the result array
# print(res)
# Extract the minimization values from the result
# minimization_values = res.fun