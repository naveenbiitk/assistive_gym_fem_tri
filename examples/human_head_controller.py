import gym, assistive_gym, argparse
import pybullet as p
import numpy as np


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



env_name = "HumanTesting-v1"
env = make_env(env_name, coop=True)
coop=True
#env = gym.make()
env.render()
observation = env.reset()
#observation= np.concatenate((observation['robot'], observation['human']))
#env.robot.print_joint_info()


# Arrow keys for moving the base, s/x for the lift, z/c for the prismatic joint, a/d for the wrist joint
#robot_actions = {p.B3G_LEFT_ARROW: np.array([0.01, -0.01, 0, 0, 0]), p.B3G_RIGHT_ARROW: np.array([-0.01, 0.01, 0, 0, 0]), p.B3G_UP_ARROW: np.array([0.01, 0.01, 0, 0, 0]), p.B3G_DOWN_ARROW: np.array([-0.01, -0.01, 0, 0, 0]), ord('s'): np.array([0, 0, 0.01, 0, 0]), ord('x'): np.array([0, 0, -0.01, 0, 0]), ord('z'): np.array([0, 0, 0, -0.01, 0]), ord('c'): np.array([0, 0, 0, 0.01, 0]), ord('a'): np.array([0, 0, 0, 0, 0.01]), ord('d'): np.array([0, 0, 0, 0, -0.01])}

human_actions = { ord('1'): np.array([ 0.01, 0, 0, 0]), ord('2'): np.array([ -0.01, 0, 0, 0]), ord('3'): np.array([ 0, 0.01, 0, 0]), ord('4'): np.array([ 0, -0.01, 0, 0]), ord('5'): np.array([ 0, 0, -0.01, 0]), ord('6'): np.array([ 0, 0, 0.01, 0]), ord('7'): np.array([ 0, 0, 0, 0.01]), ord('8'): np.array([0, 0, 0, -0.01])}





shoulder_pos,shoulder_orient = env.human.get_pos_orient(env.human.right_shoulder)
points = [0.18, -0.25, 0.1]
target_pos = [points[0]+shoulder_pos[0], points[1]+shoulder_pos[1], points[2]+shoulder_pos[2]  ]
env.create_sphere(radius=0.02, mass=0.0, pos= target_pos, visual=True, collision=False, rgba=[1, 1, 0, 1]) 

#        self.neck = 20  # 20 neck up/down pitch
#        21 head up/down pitch
#        22  right/left roll
#        23   yaw
#        self.head = 23  #--> 21 22 23 for head

        #mouth_pos, mouth_orient = p.multiplyTransforms(head_t_pos, head_t_orient, eye_pos, [0, 0, 0, 1], physicsClientId=self.id)
        #wrt_pos,wrt_orient = self.human.get_pos_orient(self.human.right_wrist)
        #self.generate_line(mouth_pos, mouth_orient)

while True:
    env.render()

    p.removeAllUserDebugItems()
    head_t_pos, head_t_orient = env.human.get_pos_orient(env.human.head)
    eye_pos = [0, -0.11, 0.1] if env.human.gender == 'male' else [0, -0.1, 0.1] # orientation needed is [0, 0, 0.707, -0.707]
    # right_ear_pos = [-0.08, -0.011, 0.08] if self.human.gender == 'male' else [-0.08, -0.01, 0.08]
    heye_pos, heye_orient = p.multiplyTransforms(head_t_pos, head_t_orient, eye_pos, [0, 0, 0, 1], physicsClientId=env.id)
    generate_line(heye_pos, heye_orient)


    human_action = np.zeros(env.action_human_len)
    keys = p.getKeyboardEvents()
    for key, a in human_actions.items():
        if key in keys and keys[key] & p.KEY_IS_DOWN:
            human_action += a

    final_action = human_action*80
    t_angle_1 = env.human.get_joint_angles([20, 21 ])
    t_angle_1 = np.array(t_angle_1)
    print('Env humanb up/down angle', t_angle_1/np.pi*180)
    # robot_action = np.zeros([1])
    # if coop:
    #     final_action =  {'robot': robot_action, 'human': human_action}
    # else:
    #     final_action = robot_action

    observation, reward, done, info = env.step(final_action)