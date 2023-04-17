import gym, assistive_gym, argparse
import pybullet as p
import numpy as np
import yaml

from assistive_gym.learn import make_env
from math import sin,cos
from PIL import Image
import math




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

        #if count_==5:
        #    img, depth = env.get_camera_image_depth()
        #    im = Image.fromarray(img)
        #    im.save(st+'.png')




env_name = "HumanTesting-v1"
env = make_env(env_name, coop=True)
coop=True
#env = gym.make()
env.render()
observation = env.reset()
#observation= np.concatenate((observation['robot'], observation['human']))
#env.robot.print_joint_info()

human_actions_keys = { ord('1'): np.array([ 0.01, 0, 0, 0]), ord('2'): np.array([ -0.01, 0, 0, 0]), ord('3'): np.array([ 0, 0.01, 0, 0]), ord('4'): np.array([ 0, -0.01, 0, 0]), ord('5'): np.array([ 0, 0, -0.01, 0]), ord('6'): np.array([ 0, 0, 0.01, 0]), ord('7'): np.array([ 0, 0, 0, 0.01]), ord('8'): np.array([0, 0, 0, -0.01])}

shoulder_pos,shoulder_orient = env.human.get_pos_orient(env.human.right_shoulder)
#points = [0.18, -0.25, 0.1]
points = [0.18, -0.4, 0.1]
#points = [-0.04050275653090826, -0.38547224068225105, -0.14691517168799585]
target_pos = [points[0]+shoulder_pos[0], points[1]+shoulder_pos[1], points[2]+shoulder_pos[2]  ]
env.create_sphere(radius=0.02, mass=0.0, pos= target_pos, visual=True, collision=False, rgba=[1, 1, 0, 1]) 


head_point = np.array(env.human.get_pos_orient(env.human.head)[0],dtype="float64")
neck_point = np.array(env.human.get_pos_orient(env.human.neck)[0],dtype="float64")
chest_point = np.array(env.human.get_pos_orient(env.human.right_pecs)[0],dtype="float64")
waist_point = np.array(env.human.get_pos_orient(env.human.waist)[0],dtype="float64")
stomach_point = np.array(env.human.get_pos_orient(env.human.stomach)[0],dtype="float64")
left_shoulder_point = np.array(env.human.get_pos_orient(env.human.left_shoulder)[0],dtype="float64")
right_shoulder_point = np.array(env.human.get_pos_orient(env.human.right_shoulder)[0],dtype="float64")
left_elbow_point = np.array(env.human.get_pos_orient(env.human.left_elbow)[0],dtype="float64")
right_elbow_point = np.array(env.human.get_pos_orient(env.human.right_elbow)[0],dtype="float64")
        
handover_point = np.array(target_pos,dtype="float64")

point_wrt_head = handover_point - head_point
point_wrt_neck = handover_point - neck_point
point_wrt_chest = handover_point - chest_point
point_wrt_waist = handover_point - waist_point
point_wrt_stomach = handover_point - stomach_point
point_wrt_left_shoulder = handover_point - left_shoulder_point
point_wrt_right_shoulder = handover_point - right_shoulder_point
point_wrt_left_elbow = handover_point - left_elbow_point
point_wrt_right_elbow = handover_point - right_elbow_point

#save the points
dict_file = [ {'point_wrt_head':[point_wrt_head.tolist()]},
              {'point_wrt_neck':[point_wrt_neck.tolist()]},
              {'point_wrt_chest':[point_wrt_chest.tolist()]},
              {'point_wrt_waist':[point_wrt_waist.tolist()]},
              {'point_wrt_stomach':[point_wrt_stomach.tolist()]},
              {'point_wrt_left_shoulder':[point_wrt_left_shoulder.tolist()]},
              {'point_wrt_right_shoulder':[point_wrt_right_shoulder.tolist()]},
              {'point_wrt_left_elbow':[point_wrt_left_elbow.tolist()]},
              {'point_wrt_right_elbow':[point_wrt_right_elbow.tolist()]}
              ]

with open(r'reference_points_showscreen.yaml', 'w') as file:
    documents = yaml.dump(dict_file, file, default_flow_style=False)



human_base_pose, human_orient = env.human.get_pos_orient(env.human.base) 
orientation = p.getEulerFromQuaternion(np.array(human_orient), physicsClientId=env.id) 
print('orientation yaw',orientation[2])

for i_ in range(-5,5,2):
    i = i_*8
    orientation_new = [orientation[0], orientation[1], orientation[2]+(i/180*3.14)]
    new_human_orient = p.getQuaternionFromEuler(orientation_new, physicsClientId=env.id)
    #env.human.set_base_pos_orient(human_base_pose, new_human_orient )
    joints_positions = [ (env.human.j_waist_z, -30)]
    #env.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None)
    run_simulation(env.human, env.id, 15)

