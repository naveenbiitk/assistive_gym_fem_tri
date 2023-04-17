import gym, assistive_gym
import numpy as np
from assistive_gym.learn import make_env
import pybullet as p
import math
from math import sin, cos


class stretch_sim_real():

    def __init__(self, env_name = "StretchTesting-v1", render=True, seed=1, num_frames=200):
        #super().__init__(render=render, seed=seed, num_frames=num_frames)

        self.env_name = env_name
        self.num_frames = num_frames
        self.env_loader()
        self.keyboard_teleop = True
        self.keys_actions = {p.B3G_LEFT_ARROW: np.array([0.01, -0.01, 0, 0, 0]), p.B3G_RIGHT_ARROW: np.array([-0.01, 0.01, 0, 0, 0]), p.B3G_UP_ARROW: np.array([0.01, 0.01, 0, 0, 0]), p.B3G_DOWN_ARROW: np.array([-0.01, -0.01, 0, 0, 0]), ord('s'): np.array([0, 0, 0.01, 0, 0]), ord('x'): np.array([0, 0, -0.01, 0, 0]), ord('z'): np.array([0, 0, 0, -0.01, 0]), ord('c'): np.array([0, 0, 0, 0.01, 0]), ord('a'): np.array([0, 0, 0, 0, 0.01]), ord('d'): np.array([0, 0, 0, 0, -0.01])}


    def env_loader(self):
        
        self.env = make_env(self.env_name)
        #env = gym.make()
        self.env.render()
        observation = self.env.reset()
        

    def keyboard_actions(self):

        action = np.zeros(self.env.action_robot_len)
        if self.keyboard_teleop:

            keys = p.getKeyboardEvents()
            for key, a in self.keys_actions.items():
                if key in keys and keys[key] & p.KEY_IS_DOWN:
                    action += a

        return action

#  <------------------------------------------------------------->

    def generate_target_point(self,pos, orient):
        lineLen=0.5
        mat = p.getMatrixFromQuaternion(orient)
        dir0 = [mat[0], mat[3], mat[6]]
        dir2_neg = [-mat[1], -mat[4], -mat[7]]
        k1 = np.random.random()+0.3
        k2 = np.random.random()+0.3
        k3 = np.random.random()+0.3
        p1 = [pos[0] + k1*lineLen * (0.9*dir2_neg[0]+0.1*dir0[0]), pos[1] + k1*lineLen * (0.9*dir2_neg[1]+0.1*dir0[1]), pos[2] + k1*lineLen * (0.9*dir2_neg[2]+0.1*dir0[2])/2 ]
        p2 = [pos[0] + k2*lineLen * (0.9*dir2_neg[0]-0.1*dir0[0]), pos[1] + k2*lineLen * (0.9*dir2_neg[1]-0.1*dir0[1]), pos[2] + k2*lineLen * (0.9*dir2_neg[2]+0.1*dir0[2])/2 ]
        pf = [(p1[0]+p2[0])/2,(p1[1]+p2[1])/2,(p1[2]+p2[2])/2]
        pf[2] = k3
    
        return pf


    def some_func(self, pf): 

        base_pos, base_orient = self.env.robot.get_pos_orient(self.env.robot.base)
        #generate_target_point(pos, orient)
        #current_joint_angles = env.robot.get_joint_angles(env.robot.left_arm_joint_indices)
        yaw_orientation = np.arctan2(2*(base_orient[3]*base_orient[2]+base_orient[0]*base_orient[1]),1-2*(base_orient[1]*base_orient[1]+base_orient[2]*base_orient[2]))
        base_pos[2] = yaw_orientation/math.pi*180
        current_joint_angles = self.env.robot.get_joint_angles(self.env.robot.left_arm_joint_indices)

        target_joint_angles = self.env.robot.ik(self.env.robot.left_tool_joint, pf, base_orient, ik_indices=self.env.robot.left_arm_ik_indices, max_iterations=200)
        #print('Target angles', target_joint_angles)
        self.env.robot.set_joint_angles(self.env.robot.left_arm_joint_indices, target_joint_angles)



    def create_sphere_base(self):
        base_pos, base_orient = self.env.robot.get_pos_orient(self.env.robot.base)
        #pf = self.generate_target_point(base_pos,base_orient)
        pf = self.env.target_pos
        yaw_orientation = np.arctan2(2*(base_orient[3]*base_orient[2]+base_orient[0]*base_orient[1]),1-2*(base_orient[1]*base_orient[1]+base_orient[2]*base_orient[2]))
        base_pos_set = [pf[0]-0.0*cos(yaw_orientation)-0.4,pf[1]-0.0*sin(yaw_orientation)-0.1,base_pos[2]]
    
        self.env.robot.set_base_pos_orient( base_pos_set, base_orient)
        self.env.create_sphere(radius=0.01, mass=0.0, pos=pf, visual=True, collision=False, rgba=[0, 0, 1, 1])
        return pf


    def env_render(self):

        pf = self.create_sphere_base()
        for i_ in range(self.num_frames):
            self.env.render()
            #self.some_func(pf)
            action = self.keyboard_actions()
            print('Action: ', action[3]*100) # wrist extension
            observation, reward, done, info = self.env.step(action)




if __name__ == "__main__":
    c = stretch_sim_real()
    c.env_render()