#!/usr/bin/env python3  

#-------#
# Node broadcasts the tf for the aruco tag on the bottom left of the bed with respect to map frame
# Node write the function name
#-------#

from __future__ import division, print_function
import rospy
import tf
import tf2_msgs.msg
import numpy as np
import time
import argparse, ray
#import geometry_msgs.msg


from std_msgs.msg import Float64MultiArray
import assistive_gym
from assistive_gym.learn import load_policy, make_env

from sensor_msgs.msg import JointState
from control_msgs.msg import FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint

import hello_helpers.hello_misc as hm
import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob
import numpy as np
# from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.agents import ppo, sac
from ray.tune.logger import pretty_print
from numpngw import write_apng




class Stretch_custom_controller(hm.HelloNode):

    def __init__(self):

        hm.HelloNode.main(self, 'Stretch_real_rl', 'Stretch_real_rl', wait_for_first_pointcloud=False)
        self.frame = '/base_link'
        self.targetFrame = '/link_grasp_center'
        self.tf = tf.TransformListener()
        self.TotalJointPositions = []
        self.LiftJointPositions = -1
        self.ArmJointPositions = -1
        self.WristJointPositions = -1
        self.controlJointPositions = np.array([0.0, 0.0, 0.0])
        self.frame_skip = 5
        hm.HelloNode.__init__(self)

        env_name = 'StretchTesting-v1'
        ray.init(num_cpus=2, ignore_reinit_error=True, log_to_driver=False)
        #env = make_env(env_name, coop=('Human' in env_name) )
        coop = 'Human' in env_name
        self.env = make_env(env_name, coop=True) if coop else gym.make(env_name)
        
        self.env_name = env_name
        policy_path = './trained_models/'
        
        self.policy,_ = load_policy(self.env, 'ppo', self.env_name, policy_path=policy_path, coop=('Human' in self.env_name), seed=1, extra_configs={})
        #self.env.disconnect()
        #self.env = None

        self.joint_sub = rospy.Subscriber("/joint_states", JointState, self.get_joint_angles, queue_size=1)
        self.action_pub = rospy.Publisher('/realrl_action', Float64MultiArray, queue_size=10)
        self.obs_pub = rospy.Publisher('/realrl_observation', Float64MultiArray, queue_size=10)
        print('Policy running')

        self.robot_obs_list = []
        self.robot_rl_action_list = []
        self.robot_point_action_list = []
        
        # now = rospy.Time.now()
        # self.tf.waitForTransform(self.frame, self.targetFrame, now, rospy.Duration(10.0))
        # Return the most revent transformation
    def get_Gripper_Position(self):
            
        position=[0,0,0]
        quaternion=[0,0,0,1]


        try:
            (position, quaternion) = self.tf.lookupTransform(self.targetFrame, self.frame,  rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            postion=[0,0,0]
            quaternion=[0,0,0,1]

        # if not Quat:
        #     return position, euler_from_quaternion(quaternion)
        return position, quaternion
       

    def get_joint_angles(self, msg):
        #self.TotalJointPositions = msg.position
        self.LiftJointPositions = msg.position[4]
        self.ArmJointPositions = msg.position[8]
        self.WristJointPositions = msg.position[9]
        #print('Joint angles',self.LiftJointPositions,self.ArmJointPositions,self.WristJointPositions)

    def get_observation(self):
        target_pos_real = [0.35, -0.2, 0.92]
        pos_1,quat = self.get_Gripper_Position()
        pos = [-pos_1[0],-pos_1[1],-pos_1[2]]
        end_effector_pos_diff = np.array(target_pos_real)-np.array(pos)
        robot_joint_angles = np.array([self.LiftJointPositions, self.ArmJointPositions, self.WristJointPositions ])
        obs = np.concatenate([target_pos_real, end_effector_pos_diff,pos,quat,robot_joint_angles ]).ravel()
        return obs

    def moveToJointAngles(self, target_joints):
        stow_point = JointTrajectoryPoint()
        stow_point.time_from_start = rospy.Duration(0.0)
        stow_point.positions = target_joints

        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.trajectory.joint_names = ['joint_lift', 'wrist_extension', 'joint_wrist_yaw']
        trajectory_goal.trajectory.points = [stow_point]
        trajectory_goal.trajectory.header.stamp = rospy.Time(0.0)
        trajectory_goal.trajectory.header.frame_id = 'base_link' 

        self.trajectory_client.send_goal(trajectory_goal)
        rospy.loginfo('Sent point action goal = {}'.format(trajectory_goal))
        self.trajectory_client.wait_for_result()


    def issue_multipoint_command(self):
        point0 = JointTrajectoryPoint()
        point0.positions = [0.2, 0.0, 3.4]
        point0.velocities = [0.2, 0.2, 2.5]
        point0.accelerations = [1.0, 1.0, 3.5]

        point1 = JointTrajectoryPoint()
        point1.positions = [0.3, 0.1, 2.0]

        point2 = JointTrajectoryPoint()
        point2.positions = [0.5, 0.2, -1.0]

        point3 = JointTrajectoryPoint()
        point3.positions = [0.6, 0.3, 0.0]

        point4 = JointTrajectoryPoint()
        point4.positions = [0.8, 0.2, 1.0]

        point5 = JointTrajectoryPoint()
        point5.positions = [0.5, 0.1, 0.0]

        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.trajectory.joint_names = ['joint_lift', 'wrist_extension', 'joint_wrist_yaw']
        trajectory_goal.trajectory.points = [point0, point1, point2, point3, point4, point5 ]
        trajectory_goal.trajectory.header.stamp = rospy.Time(0.0)
        trajectory_goal.trajectory.header.frame_id = 'base_link' 

        self.trajectory_client.send_goal(trajectory_goal)
        rospy.loginfo('Sent stow goal = {0}'.format(trajectory_goal))
        self.trajectory_client.wait_for_result()





        
        #p.setJointMotorControlArray(self.body, jointIndices=indices, controlMode=p.POSITION_CONTROL, targetPositions=target_angles, positionGains=gains, forces=forces, physicsClientId=self.id)


    def initalize(self):
        #init_joint_angles = np.array([0.77725216, 0.0,  0.0])

        stow_point = JointTrajectoryPoint()
        stow_point.time_from_start = rospy.Duration(0.0)
        stow_point.positions = np.array([0.77725216, 0.0,  0.0])

        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.trajectory.joint_names = ['joint_lift', 'wrist_extension', 'joint_wrist_yaw']
        trajectory_goal.trajectory.points = [stow_point]
        trajectory_goal.trajectory.header.stamp = rospy.Time(0.0)
        trajectory_goal.trajectory.header.frame_id = 'base_link' 

        self.trajectory_client.send_goal(trajectory_goal)
        rospy.loginfo('Sent point action goal = {}'.format(trajectory_goal))
        self.trajectory_client.wait_for_result()



    def real_rl(self):

        
        self.initalize()
        #pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        rate = rospy.Rate(10)
        action_arr = Float64MultiArray()
        obs_arr = Float64MultiArray()
        
        observation_ = self.env.reset()

        joint_sim_action_fix = np.array([0.68906046, 0.0  ,0.0   ,0.0  ,0.0, 0.03806068])
        joint_sim_action_delta = np.array([0.00, 0.0  ,0.0   ,0.0  ,0.00, 1.8])
        count_=0
        count_threshold = 10
        while not rospy.is_shutdown() and count_<count_threshold+1:

            obs = self.get_observation()
            
            if 'Human' in self.env_name:
                action = self.policy.compute_action(obs, policy_id='robot')
            else:
                action = self.policy.compute_action(obs)

            action = np.clip(action, a_min=self.env.action_space.low, a_max=self.env.action_space.high)
            action *= self.env.robot.action_multiplier

            joint_sim_action = self.env.take_step_robot( action, gains=None, forces=None, action_multiplier=0.05, step_sim=True)

            #joint_sim_action_fix = joint_sim_action_fix+joint_sim_action_delta
            if count_>3:
                joint_sim_action_fix = joint_sim_action+joint_sim_action_delta
            else:
                joint_sim_action_fix = joint_sim_action

            #print('Action computed', joint_sim_action_fix)
            print('Iteration', count_)
            #self.scaled_action = np.clip(np.array(action), -0.2, 0.2)
            #target_joints = self.controlJointPositions + self.scaled_action*self.frame_skip
            #target_joints = np.array([joint_sim_action[0],0,joint_sim_action[5]])
            target_joints = np.array([joint_sim_action_fix[0],joint_sim_action_fix[4],joint_sim_action_fix[5]])
            self.moveToJointAngles(target_joints)

            action_arr.data = action
            obs_arr.data = obs
            self.action_pub.publish(action_arr)
            self.obs_pub.publish(obs_arr)

            #print('Observation',obs, ' action',action)

            self.robot_obs_list.append(obs)
            self.robot_rl_action_list.append(joint_sim_action)
            #print(obs)
            if count_==count_threshold:
                print('RL completed-------------------------------')
                print('Saved-------------------------------')
                np.save('robot_observation_space_joint.npy', np.array(self.robot_obs_list) )
                np.save('robot_action_space_joint.npy', np.array(self.robot_rl_action_list) )
                #save
            count_ = count_+1
            rate.sleep()




if __name__ == '__main__':
    #rospy.init_node('Stretch_real_rl', anonymous=True)

    sth = Stretch_custom_controller()
    sth.real_rl()
