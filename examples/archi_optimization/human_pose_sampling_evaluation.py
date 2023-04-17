import gym, assistive_gym, argparse
import pybullet as p
import numpy as np
import yaml

from assistive_gym.learn import make_env
from math import sin,cos
from PIL import Image
import math

#import cmaes_handover 
import cmaes_showphone


index = 0



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



if __name__ == "__main__":

    env_name = "HumanTesting-v1"
    env = make_env(env_name, coop=True)
    coop=True
    #env = gym.make()
    #env.render()
    env.setup_camera()
    observation = env.reset()
    #observation= np.concatenate((observation['robot'], observation['human']))
    #env.robot.print_joint_info()
    #print('----------envhumanlen--',env.action_human_len)
    human_actions_keys = { ord('1'): np.array([ 0.01, 0, 0, 0]), ord('2'): np.array([ -0.01, 0, 0, 0]), ord('3'): np.array([ 0, 0.01, 0, 0]), ord('4'): np.array([ 0, -0.01, 0, 0]), ord('5'): np.array([ 0, 0, -0.01, 0]), ord('6'): np.array([ 0, 0, 0.01, 0]), ord('7'): np.array([ 0, 0, 0, 0.01]), ord('8'): np.array([0, 0, 0, -0.01])}

    shoulder_pos,shoulder_orient = env.human.get_pos_orient(env.human.right_shoulder)
    points = [0.18, -0.25, 0.1]
    target_pos = [points[0]+shoulder_pos[0], points[1]+shoulder_pos[1], points[2]+shoulder_pos[2]  ]
    #env.create_sphere(radius=0.02, mass=0.0, pos= target_pos, visual=True, collision=False, rgba=[1, 1, 0, 1]) 


    # self.j_waist_x, self.j_waist_y, self.j_waist_z = 25, 26, 27
    # joints_positions = [(env.human.j_head_x, i), (env.human.j_head_y, j), (env.human.j_head_z, k)]
    # y is not useful at all waist x,z  x would be useful from -3  waist is moving the bottom body alone not useful
    # orientation_new = [orientation[0], orientation[1], orientation[2]+(i/180*3.14)]
    # new_human_orient = p.getQuaternionFromEuler(orientation_new, physicsClientId=env.id)
    # env.human.set_base_pos_orient(human_base_pose, new_human_orient )
    # useful in rotating the whole human
    # self.j_left_pecs_x, self.j_left_pecs_y, self.j_left_pecs_z 


    human_base_pose, human_orient = env.human.get_pos_orient(env.human.base) 
    orientation = p.getEulerFromQuaternion(np.array(human_orient), physicsClientId=env.id) 
    print('orientation yaw',orientation[2])
                
    with open("reference_points_showscreen.yaml", 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    print(data_loaded)

    point_wrt_head = (data_loaded[0]['point_wrt_head'])[0]
    point_wrt_neck = (data_loaded[1]['point_wrt_neck'])[0]
    point_wrt_chest = (data_loaded[2]['point_wrt_chest'])[0]
    point_wrt_waist = (data_loaded[3]['point_wrt_waist'])[0]
    point_wrt_stomach = (data_loaded[4]['point_wrt_stomach'])[0]
    point_wrt_left_shoulder = (data_loaded[5]['point_wrt_left_shoulder'])[0]
    point_wrt_right_shoulder = (data_loaded[6]['point_wrt_right_shoulder'])[0]
    point_wrt_left_elbow = (data_loaded[7]['point_wrt_left_elbow'])[0]
    point_wrt_right_elbow = (data_loaded[8]['point_wrt_right_elbow'])[0]

    target_points = np.zeros([9,3])

    #print('target_points', target_points)
    total_index=256
    evaluation_arr = np.zeros([total_index, 9])


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
                    #run_simulation(env.human, env.id, 10)
                    head_point = np.array(env.human.get_pos_orient(env.human.head)[0],dtype="float64")
                    neck_point = np.array(env.human.get_pos_orient(env.human.neck)[0],dtype="float64")
                    chest_point = np.array(env.human.get_pos_orient(env.human.right_pecs)[0],dtype="float64")
                    waist_point = np.array(env.human.get_pos_orient(env.human.waist)[0],dtype="float64")
                    stomach_point = np.array(env.human.get_pos_orient(env.human.stomach)[0],dtype="float64")
                    left_shoulder_point = np.array(env.human.get_pos_orient(env.human.left_shoulder)[0],dtype="float64")
                    right_shoulder_point = np.array(env.human.get_pos_orient(env.human.right_shoulder)[0],dtype="float64")
                    left_elbow_point = np.array(env.human.get_pos_orient(env.human.left_elbow)[0],dtype="float64")
                    right_elbow_point = np.array(env.human.get_pos_orient(env.human.right_elbow)[0],dtype="float64")

                    target_points[0] = head_point + point_wrt_head
                    target_points[1] = neck_point + point_wrt_neck
                    target_points[2] = chest_point + point_wrt_chest
                    target_points[3] = waist_point + point_wrt_waist
                    target_points[4] = stomach_point + point_wrt_stomach
                    target_points[5] = left_shoulder_point + point_wrt_left_shoulder
                    target_points[6] = right_shoulder_point + point_wrt_right_shoulder
                    target_points[7] = left_elbow_point + point_wrt_left_elbow
                    target_points[8] = right_elbow_point + point_wrt_right_elbow

                    for r in range(9):
                        evaluation_point = target_points[r]
                        #evaluation_arr[index][r] = human_show_phone_controller.optimizing_function(evaluation_point, env.human, env.id, env )
                        evaluation_arr[index][r] = cmaes_showphone.optimizing_function(evaluation_point, env.human, env.id, env) #evaluate_object_handover(evaluation_point)

                    index=index+1
                    print('At index ', index)

    outfile='evaluation_showscreen.npy'
    np.save(outfile, evaluation_arr )