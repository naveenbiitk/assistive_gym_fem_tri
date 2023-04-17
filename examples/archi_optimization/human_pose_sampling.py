import gym, assistive_gym, argparse
import pybullet as p
import numpy as np


from assistive_gym.learn import make_env
from math import sin,cos
from PIL import Image
import math
import time

index = 0

def generate_line(pos, orient, lineLen=0.5):
        
    #p.removeAllUserDebugItems()
    mat = p.getMatrixFromQuaternion(orient)
    dir0 = [mat[0], mat[3], mat[6]]
    dir1 = [mat[1], mat[4], mat[7]]
    dir2 = [mat[2], mat[5], mat[8]]
    
    # works only for hand 0.25 linelen
    #dir2_neg = [-mat[2], -mat[5], -mat[8]]
    #to1 = [pos[0] + lineLen * (dir2_neg[0]+dir0[0])/2, pos[1] + lineLen * (dir2_neg[1]+dir0[1])/2, pos[2] + lineLen * (dir2_neg[2]+dir0[2])/2 ]
    #to2 = [pos[0] + lineLen * (dir2_neg[0]-dir0[0])/2, pos[1] + lineLen * (dir2_neg[1]-dir0[1])/2, pos[2] + lineLen * (dir2_neg[2]+dir0[2])/2 ]
    
    # works only for head  1.5 linlen
    dir2_neg = [-mat[1], -mat[4], -mat[7]]
    to1 = [pos[0] + lineLen * (dir2_neg[0]+dir0[0])/2, pos[1] + lineLen * (dir2_neg[1]+dir0[1])/2, pos[2] + lineLen * (dir2_neg[2]+dir0[2])/2 ]
    to2 = [pos[0] + lineLen * (dir2_neg[0]-dir0[0])/2, pos[1] + lineLen * (dir2_neg[1]-dir0[1])/2, pos[2] + lineLen * (dir2_neg[2]+dir0[2])/2 ]
    
    toX = [pos[0] + lineLen * dir0[0], pos[1] + lineLen * dir0[1], pos[2] + lineLen * dir0[2]]
    toY = [pos[0] + lineLen * dir1[0], pos[1] + lineLen * dir1[1], pos[2] + lineLen * dir1[2]]
    toZ = [pos[0] + lineLen * dir2[0], pos[1] + lineLen * dir2[1], pos[2] + lineLen * dir2[2]]
    
    #p.addUserDebugLine(pos, toX, [1, 0, 0], 5)
    toY = [pos[0] - lineLen * dir1[0], pos[1] - lineLen * dir1[1], pos[2] - lineLen * dir1[2]]
    p.addUserDebugLine(pos, toY, [1, 0, 0], 5)
    #p.addUserDebugLine(pos, toZ, [0, 0, 1], 5)




def run_simulation(env_human, env_id, count_target):
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
        generate_line(heye_pos, heye_orient)

         # t = p.addUserDebugText(text=st, textPosition=[-0.8,0,1], textColorRGB=[1,1,1], textSize=2, lifeTime=0.5, physicsClientId=env.id )
        human_action = np.zeros(env.action_human_len)
        
        observation, reward, done, info = env.step(human_action)
        count_ = count_ + 1

        if count_==5:
           img, depth = env.get_camera_image_depth()
           im = Image.fromarray(img)
           st = str(index)
           im.save('human_sample/'+st+'.png')




env_name = "HumanTesting-v1"
env = make_env(env_name, coop=True)
coop=True
#env = gym.make()
env.render()
env.setup_camera()
observation = env.reset()
#observation= np.concatenate((observation['robot'], observation['human']))
#env.robot.print_joint_info()

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
# self.set_base_pos_orient([0, 0.03, 0.89 if self.gender == 'male' else 0.86], [0, 0, 0, 1])

human_base_pose, human_orient = env.human.get_pos_orient(env.human.base) 
orientation = p.getEulerFromQuaternion(np.array(human_orient), physicsClientId=env.id) 
print('orientation yaw',orientation[2])




# for h_i_ in range(-3,4,2):
#     for h_j_ in range(-3,4,2):
#         for j_ in range(-3,4,2):
#             for i_ in range(-3,4,2):
#                 i = i_*10
#                 j = j_*10
#                 h_i = h_i_*10
#                 h_j = h_j_*10
#                 orientation_new = [orientation[0], orientation[1], orientation[2]+(i/180*3.14)]
#                 new_human_orient = p.getQuaternionFromEuler(orientation_new, physicsClientId=env.id)
#                 env.human.set_base_pos_orient(human_base_pose, new_human_orient )
#                 joints_positions = [ (env.human.j_waist_z, j), (env.human.j_head_x, h_i), (env.human.j_head_z, h_j)]# -30 to +30
#                 env.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None)
#                 run_simulation(env.human, env.id, 10)
#                 index=index+1
            





for count_ in range(50):
    h_i_ = np.random.random_integers(-3, 4)
    h_j_ = np.random.random_integers(-3, 4)
    j_ = np.random.random_integers(-3, 4)
    i_ = np.random.random_integers(-3, 4)
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
    # for r in range(6):
    #     evaluation_point = target_points[r]
    #     evaluation_arr[index][r] = evaluate_object_handover(evaluation_point)

    index=index+1
    time.sleep(1)





# for h_i_ in range(-3,4,2):   #head_x
#     for h_j_ in range(-3,4,2):  #head_z
#         for j_ in range(-3,4,2):   # waist orientation
#             for i_ in range(-3,4,2):  # base orientation
#                 i = i_*10
#                 j = j_*10
#                 h_i = h_i_*10
#                 h_j = h_j_*10
#                 orientation_new = [orientation[0], orientation[1], orientation[2]+(i/180*3.14)]
#                 new_human_orient = p.getQuaternionFromEuler(orientation_new, physicsClientId=env.id)
#                 env.human.set_base_pos_orient(human_base_pose, new_human_orient )
#                 joints_positions = [ (env.human.j_waist_z, j), (env.human.j_head_x, h_i), (env.human.j_head_z, h_j)]# -30 to +30
#                 env.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None)
#                 #run_simulation(env.human, env.id, 10)
#                 # for r in range(6):
#                 #     evaluation_point = target_points[r]
#                 #     evaluation_arr[index][r] = evaluate_object_handover(evaluation_point)

#                 index=index+1
#                 time.sleep(2)


print('Final index',index)