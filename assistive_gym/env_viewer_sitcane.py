import gym, sys, argparse
import numpy as np
import pybullet as p
from assistive_gym.learn import make_env
# import assistive_gym
from numpngw import write_apng
import cv2
from PIL import Image 
import time
import os
import pickle

if sys.version_info < (3, 0):
    print('Please use Python 3')
    exit()

def sample_action(env, coop):
    # if coop:
    # time.sleep(5)
    #     return {'robot': env.action_space_robot.sample(), 'human': env.action_space_human.sample()}
    return env.action_space.sample()

def sample_action_human(env, coop):
    #if coop:
    #    return {'robot': env.action_space_robot.sample(), 'human': env.action_space_human.sample()}
    return env.action_space_human.sample()


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


def viewer(env_name):
    coop = 'Human' in env_name
    env = make_env(env_name, coop=True) if coop else gym.make(env_name)
    env.render()
    #observation = env.reset()
    #observation = env.reset()
    sample_i=1
    frames = []
    
    pkl_file_name = '/nethome/nnagarathinam6/hrl_git/assistive-gym-fem/examples/optimal_frame_lying/data/smpl_pkl_35/smpl_smpl_postures74_35.pkl'
    env.set_file_name(pkl_file_name)
    
    dir_base = '/nethome/nnagarathinam6/hrl_git/assistive-gym-fem/examples/optimal_frame_lying/'
    dir_files = os.listdir(dir_base+'data/')

    dir_file_name = dir_base + 'data/' + dir_files[3]
    pkl_files = os.listdir(dir_file_name)

    while sample_i<2:
        observation = env.reset()
        #env.render()
        done = False
        # action = sample_action(env, coop)
        action = sample_action(env, coop)
        # if coop:
        #     print('Robot observation size:', np.shape(observation['robot']), 'Human observation size:', np.shape(observation['human']), 'Robot action size:', np.shape(action['robot']), 'Human action size:', np.shape(action['human']))
        if coop:
            print('Observation: ',observation)
        elif 'BeddingManipulationSphere-v1' in env_name:
            action = np.array([0.3, 0.5, 0, 0])
        elif 'RemoveContactSphere-v1' in env_name:
            action = np.array([0.3, 0.45])
        else:
            print('Observation size:', np.shape(observation), 'Action size:', np.shape(action))

        count_=0
        while count_<80:
            # observation, reward, done, info = env.step(sample_action(env, coop))
            # action = np.zeros([5])
            observation, reward, done, info = env.step(action)
            count_=count_+1
            # if coop:
            #     done = done['__all__']

        #img, depth = env.get_camera_image_depth()
        #frame_ = int(env.sample_pkl)-1

        #pkl_file_name = dir_file_name +'/'+ pkl_files[sample_i]
        #env.set_file_name(pkl_file_name)
        base_pos, transform_orient = env.tool.get_base_pos_orient()
        transform_pos = np.array([base_pos[0], base_pos[1], base_pos[2]])
        offset_pos = np.array([-0.2,0.6,-0.15]) 
        print('transform pose', transform_pos+offset_pos)

        transform_pos = [-0.495525317, -0.239432083,  0.36575483] #correct pose
        #transform_pos = [-0.195525317, 0.239432083,  0.36575483]
        transform_orient = [0,0,0,1]
        env.tool.set_base_pos_orient(transform_pos, transform_orient)

        collision_objects = [env.furniture, env.human]
        dists_list = []

        for obj in collision_objects:
            dists = env.tool.get_closest_points(obj, distance=0)[-1]
            dists_list.append(dists)
            print('obj ', obj, ' return ',dists )

        print('collision dist list', dists_list)

        show_bounds(env, arm_length=0.5, bnd_point = transform_pos)
        #env.change_human_pose()
        #cv2.imwrite(filename, img)
        #pos_offset = [0.15,0,0.5] #this is for constraint
        #orient_offset = env.get_quaternion([0,0,np.pi])
        #sphere_pos, sphere_orient = p.multiplyTransforms(positionA=transform_pos, orientationA=transform_orient, positionB=pos_offset, orientationB=orient_offset, physicsClientId=env.id)
            
        #target_position = sphere_pos
        #env.target = env.create_sphere(radius=0.02, mass=0.0, pos= target_position, visual=True, collision=False, rgba=[1, 1, 0, 1]) 
        #target_joint_angles_h, _ = position_human_arm(env, target_position, attempts=40)
        #env.human.set_joint_angles(env.human.right_arm_joints, target_joint_angles_h)
        #filename = 'smpl_pybullet_img_25/smpl_postures'+str(frame_)+'_25.png'
        #cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #frames.append(img)
        time.sleep(3)
        sample_i += 1
    

def optimizing_across_poses(point, env, frame_i):

    observation = env.reset()

    human_base_pose, human_orient = env.human.get_pos_orient(env.human.base) 
    orientation = p.getEulerFromQuaternion(np.array(human_orient), physicsClientId=env.id) 

    index = 0
    total_index=256
    score_array = np.zeros([total_index])

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
                    #print('Evaluation point: ',evaluation_point)
                    score_array[index] = cmaes_showphone.optimizing_function(evaluation_point, env.human, env.id, env) #evaluate_object_handover(evaluation_point)
                    index=index+1
                    #print('At index ', index)

    f_value = np.sum(score_array)
    print('Iteration done: f_value: ', f_value)
    return f_value    
    


def show_bounds(env, arm_length, bnd_point):

    bnd_point = [-0.495525317, -0.239432083,  0.11575483] 
    point=np.array([0,0,0])

    x_min = bnd_point[0]- 0.0*arm_length 
    x_max = bnd_point[0]+ 0.5*arm_length

    y_min = bnd_point[1] - 0.4*arm_length
    y_max = bnd_point[1] + 0.6*arm_length

    #z_min = bnd_point[2] + 0.2*arm_length
    #z_max = bnd_point[2] + 1.0*arm_length

    z_min = bnd_point[2] + 0.5*arm_length
    z_max = bnd_point[2] + 1.5*arm_length

    x_ = np.linspace(x_min,x_max,12)
    y_ = np.linspace(y_min,y_max,10)
    z_ = np.linspace(z_min,z_max,8)

    X, Y, Z = np.meshgrid(x_ ,y_, z_) 

    x=X.flatten()
    y=Y.flatten()
    z=Z.flatten()

    points_sphere = np.array([x,y,z])
    points_sphere = points_sphere.transpose()

    env.create_spheres(radius=0.01, mass=0.0, batch_positions=points_sphere, visual=True, collision=False, rgba=[1, 1, 0, 1])

    sphere_collision = -1
    radius=0.01
    mass=0.0
    rgba=[0, 1, 1, 1]
    sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=rgba, physicsClientId=env.id)
    maximal_coordinates=False 
    collision_objects = [env.furniture]

    return

    score_list = []
    collision_list = []
    tool_pos_list = []
    count_ = 0
    for i in x_:
        for j in y_:
            for k in z_:
                
                count_ = count_+1
                if count_!=409:
                    continue

                transform_pos = [i+0.1,j,k+0.5]

                tool_pos = [i,j,k]
                transform_orient = [0,0,0,1]
                env.tool.set_base_pos_orient(tool_pos, transform_orient)
                
                env.create_sphere(radius=0.02, mass=0.0, pos= transform_pos, visual=True, collision=False, rgba=[1, 1, 0, 1])
                dists_list = []
                for obj in collision_objects:
                    dists = env.tool.get_closest_points(obj, distance=0)[-1]
                    dists_list.append(dists)
                #     #print('obj ', obj, ' return ',dists )
                collision_flag = True
                if all(not d for d in dists_list):
                    collision_flag = False
                #     #print('----------Collision detected-------')

                target_joint_angles_h, score = position_human_arm(env, transform_pos, attempts=80)
                env.human.set_joint_angles(env.human.right_arm_joints, target_joint_angles_h)
                
                print('count_', count_, 'score', score, 'collision_flag', collision_flag)
                score_list.append(score)
                collision_list.append(collision_flag)
                tool_pos_list.append(tool_pos)


    score_array = np.array(score_list)
    collision_array = np.array(collision_list)
    tool_pos_array = np.array(tool_pos_list)
    
    # dict_item = { 'Score_points': score_array,'collision_flag' : collision_array , 'tool_pos': tool_pos_array }
    # f = open("file_modified_cane_set1.pkl","wb")
    # pickle.dump(dict_item,f)
    # f.close()
    
'''
make the robot stiff
for loop of the cane see
    1. set the base-orient  
    env.tool.set_base_pos_orient(transform_pos, transform_orient)
    
    2. collision detection
    see whether cane is colliding with the chair and human

    3. ik/ score for reaching the point
    the function is there object handover 

    4. Score that in list
    save the score

    5. What to plot one of the good candidate is low score

    6. What about frames/ How to make connections?
    frames/how to make connections
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assistive Gym Environment Viewer')
    parser.add_argument('--env', default='HumanSitcane-v1',
                        help='Environment to test (default: ScratchItchJaco-v1)')
    args = parser.parse_args()

    viewer(args.env)
