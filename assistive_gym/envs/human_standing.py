import os
from gym import spaces
import numpy as np
import pybullet as p

from .env import AssistiveEnv
from .agents import human
from .agents.human import Human
from .agents.human_mesh import HumanMesh
import pybullet_data

import time
human_controllable_joint_indices = human.human_walk  #human.right_arm_joints 
#human_controllable_joint_indices = human.right_arm_joints  #human.head_joints  #head_joints torso_joints


class HumanStandingEnv(AssistiveEnv):
    def __init__(self):
        super(HumanStandingEnv, self).__init__(robot=None, human=Human(human_controllable_joint_indices, controllable=True), task='human_standing', obs_robot_len= (0), obs_human_len=(16))


    def step(self, action):
        if self.human.controllable:
            h_action = action*20
            #print('--------action---------')
            #action = np.concatenate([action['robot'], action['human']])
        
        #print('----action-------',action)
        #h_action = action['human']
        self.take_step(h_action)

        obs = self._get_obs()

        end_effector_velocity = np.linalg.norm(self.human.get_velocity(self.human.right_wrist))
        
        wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]

        dist_gap = np.linalg.norm( np.array(wrist_pos)- np.array(self.target_pos) )
        
        done = self.iteration >= 100
        if dist_gap > 0.08:
            reward_abs = -0.5
            reward_time = -self.iteration
        else:
            reward_abs = +500000
            reward_time = 0
            done = True

        
        reward_action = -np.linalg.norm(action)

        reward = -self.config('distance_gap')*dist_gap + self.config('reward_abs')*reward_abs + self.config('time_w')*reward_time + self.config('action_w')*reward_action

        self.total_reward = self.total_reward+reward
        
        # if self.gui and self.iteration > 0:
            # print('Task success:', self.task_success, 'Tool force at target:', self.tool_force_at_target, reward_force_scratch)
            # print('Iteration:',self.iteration,'Task reward:', self.total_reward)
            # print('Current Reward: ', reward)

        
        info = {'total_force_on_human': self.total_reward}        #{'robot': reward, 'human': reward}
        return obs, reward,  done, info


    def _get_obs(self, agent=None):

        human_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
        
        wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]
        wrist_pos_human, _ = self.human.convert_to_realworld(wrist_pos)

        elbow_pos, elbow_orient_quat = self.human.get_pos_orient(self.human.right_elbow)
        elbow_pos_human, _ = self.human.convert_to_realworld(elbow_pos)
 
        human_obs = np.concatenate([np.array([self.iteration]),self.target_pos, human_joint_angles, wrist_pos_human, elbow_pos_human]).ravel()

        #print(human_obs)
        human_obs = human_obs.astype('float32')
        return human_obs
        robot_obs = []
        if self.human.controllable:
            return {'robot': robot_obs, 'human': human_obs}

        return {'robot': robot_obs, 'human': human_obs}



    def human_pybullet(self):
        return
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        obUids = p.loadMJCF("mjcf/humanoid.xml")
        self.humanoid = obUids[1]
        gravId = p.addUserDebugParameter("gravity", -10, 10, -10)
        jointIds = []
        paramIds = []

        p.setPhysicsEngineParameter(numSolverIterations=10)
        p.changeDynamics(self.humanoid, -1, linearDamping=0, angularDamping=0)

        for j in range(p.getNumJoints(self.humanoid)):
            p.changeDynamics(self.humanoid, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.humanoid, j)
            #print(info)
            jointName = info[1]
            jointType = info[2]
            if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
                jointIds.append(j)
                paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -4, 4, 0))

        p.setRealTimeSimulation(1)

        j_=0
        while j_<100:
            j_ += 1
            #p.setGravity(0, 0, p.readUserDebugParameter(gravId))
            for i in range(len(paramIds)):
                c = paramIds[i]
                targetPos = p.readUserDebugParameter(c)
                p.setJointMotorControl2(self.humanoid, jointIds[i], p.POSITION_CONTROL, targetPos, force=5 * 240.)
                time.sleep(0.1)







    def reset(self):
        
        super(HumanStandingEnv, self).reset()
        #p.connect(p.GUI)
        #self.build_assistive_env(furniture_type=None, human_impairment='none', fixed_human_base=False) # Human will fall down
        self.build_assistive_env(furniture_type='wheelchair', human_impairment='none', fixed_human_base=False) # Human base fix
        self.furniture.set_on_ground()
        self.furniture.set_friction(self.furniture.base, friction=5)
        # Set joint angles for human joints (in degrees)
        joints_positions = []

        human_euler = np.array([0,0,0])
        orient_human = p.getQuaternionFromEuler(human_euler, physicsClientId=self.id)
        
        human_height, human_base_height = self.human.get_heights()
        self.human.set_base_pos_orient([-0.002, -0.04, human_base_height-0.15], orient_human)
        joints_positions = [(self.human.j_right_elbow, -90), (self.human.j_left_elbow, -90), (self.human.j_right_hip_x, -90), (self.human.j_right_knee, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee, 80)]
        joints_positions += [(self.human.j_waist_x, +10)]
        joints_positions += [(self.human.j_head_x, -self.np_random.uniform(0, 10)), (self.human.j_head_y, -self.np_random.uniform(0, 10)), (self.human.j_head_z, -self.np_random.uniform(0, 10))]
        self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=None)

        print('Human height:', human_height, 'm')
        #self.human.set_base_pos_orient([0, 0.5, human_base_height], orient_human)

        p.changeDynamics(self.human.body, -1, linearDamping = 0, angularDamping = 0)
        p.changeDynamics(self.human.body, -1, lateralFriction = 0.9)

        self.point = self.create_sphere(radius=0.01, mass=0.0, pos=[0, 0, human_height], visual=True, collision=False, rgba=[0, 1, 1, 1])

        p.setGravity(0, 0, -9.81, physicsClientId=self.id)

        joints_positions = []
        self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None)
        p.resetDebugVisualizerCamera(cameraDistance=1.6, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, human_height/2.0], physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        self.target_pos = np.array([+0.016, -0.25, 1.23])
        self.init_env_variables()
        self.total_reward=0
        #self.human_pybullet()

        p.setRealTimeSimulation(1)
        gravId = p.addUserDebugParameter("gravity", -10, 10, -10)
        p.setPhysicsEngineParameter(numSolverIterations=10)
        p.changeDynamics(self.human.body, -1, linearDamping=0, angularDamping=0)
        self.jointIds = []
        self.paramIds = []

        for joints_j in self.human.controllable_joint_indices:
            self.human.enable_force_torque_sensor(joints_j) 

        for j in self.human.controllable_joint_indices:
            p.changeDynamics(self.human.body, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.human.body, j)
            print(info)
            jointName = info[1]
            jointType = info[2]
            self.jointIds.append(j)
            self.paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -40, 40, 0))

            # if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
            #     jointIds.append(j)
            #     self.paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -4, 4, 0))

        p.setRealTimeSimulation(1)


        return self._get_obs()


    def generate_target(self):

        
        
        if self.human.gender == 'male':
            self.limb, length, radius = self.human.right_elbow, 0.157, 0.033
        else:
            self.limb, length, radius = self.human.right_elbow, 0.134, 0.027
           
        self.target_on_arm = self.util.point_on_capsule(p1=np.array([0, 0, -2*length]), p2=np.array([0, 0, -length*2.5]), radius=radius, theta_range=(0, np.pi*2))
        arm_pos, arm_orient = self.human.get_pos_orient(self.limb)
        target_pos_1, target_orient = p.multiplyTransforms(arm_pos, arm_orient, self.target_on_arm, [0, 0, 0, 1], physicsClientId=self.id)

        target_pos_2 = np.array([+0.015, -0.25, 1.23])
        #array([[0.28766633, 0.96236509, 0.32128926]])
        #rnd_array = np.random.rand(3)/5  easy to train
        rnd_array = np.random.rand(3)/3  # difficult to train
        target_pos_2 = target_pos_2-rnd_array
        self.target = self.create_sphere(radius=0.02, mass=0.0, pos=target_pos_2, visual=True, collision=False, rgba=[1, 1, 0, 1]) #human hand
        #print('Target pose 1', target_pos_1)
        
        self.target_pos = np.array(target_pos_2) 
        print('---Target_pose  nooooo-----', self.target_pos)
        self.create_sphere(radius=0.02, mass=0.0, pos=target_pos_2, visual=True, collision=False, rgba=[1, 1, 0, 1])# pink robot,cup
        
        self.update_targets()



    def update_targets(self):
        return
        arm_pos, arm_orient = self.human.get_pos_orient(self.limb)
        target_pos_1, target_orient = p.multiplyTransforms(arm_pos, arm_orient, self.target_on_arm, [0, 0, 0, 1], physicsClientId=self.id)
        self.target_pos_1 = np.array(target_pos_1)
        self.target_orient = np.array(target_orient)
        self.target.set_base_pos_orient(self.target_pos, [0, 0, 0, 1])
        #print('Handover pose in main environment ',self.target_pos)
        # target should be wrt hip