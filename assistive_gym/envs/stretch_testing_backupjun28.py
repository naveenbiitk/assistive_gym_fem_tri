import numpy as np
import pybullet as p

from .env import AssistiveEnv
from .agents import furniture
from .agents.furniture import Furniture
from .agents.stretch import Stretch
from math import sin,cos
import time
from .agents.agent import Agent
from .agents.human import Human
from .agents.robot import Robot
from .agents import human
robot_arm = 'left'

# class ScratchItchStretchEnv(ScratchItchEnv):
#     def __init__(self):
#         super(ScratchItchStretchEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=False))


class StretchTestingEnv(AssistiveEnv):

    def __init__(self):
        super(StretchTestingEnv, self).__init__(robot=Stretch(robot_arm), human=Human(human.right_arm_joints, controllable=False), task='stretch_testing', obs_robot_len=16, obs_human_len=0)
        self.robot_obs_list = []
        self.robot_action_list = []
        #self.last_sim_time=None

    def take_step_robot(self, actions, gains=None, forces=None, action_multiplier=0.05, step_sim=True):
        if gains is None:
            gains = [a.motor_gains for a in self.agents]
        elif type(gains) not in (list, tuple):
            gains = [gains]*len(self.agents)
        if forces is None:
            forces = [a.motor_forces for a in self.agents]
        elif type(forces) not in (list, tuple):
            forces = [forces]*len(self.agents)
        if self.last_sim_time is None:
            self.last_sim_time = time.time()
        self.iteration += 1
        self.forces = []
        actions = np.clip(actions, a_min=self.action_space.low, a_max=self.action_space.high)
        actions *= action_multiplier
        action_index = 0
       
        for i, agent in enumerate(self.agents):
            needs_action = not isinstance(agent, Human) or agent.controllable
            if needs_action:
                agent_action_len = len(agent.controllable_joint_indices)
                
                action = np.copy(actions[action_index:action_index+agent_action_len])
                action_index += agent_action_len
                if isinstance(agent, Robot):
                    if agent.gripper_included:
                        action_name = int(actions[4]>0)
                    action *= agent.action_multiplier
                    
            # Append the new action to the current measured joint angles
            agent_joint_angles = agent.get_joint_angles(agent.controllable_joint_indices)
            # Update the target robot/human joint angles based on the proposed action and joint limits
            for _ in range(self.frame_skip):#frame_skip is 5
                if needs_action:
                    below_lower_limits = agent_joint_angles + action < agent.controllable_joint_lower_limits
                    above_upper_limits = agent_joint_angles + action > agent.controllable_joint_upper_limits
                    action[below_lower_limits] = 0
                    action[above_upper_limits] = 0
                    agent_joint_angles[below_lower_limits] = agent.controllable_joint_lower_limits[below_lower_limits]
                    agent_joint_angles[above_upper_limits] = agent.controllable_joint_upper_limits[above_upper_limits]
                    #print(loop include in stretch testing)
                if isinstance(agent, Human) and agent.impairment == 'tremor':
                    if needs_action:
                        agent.target_joint_angles += action
                    agent_joint_angles = agent.target_joint_angles + agent.tremors * (1 if self.iteration % 2 == 0 else -1)
                else:
                    agent_joint_angles += action
                    if isinstance(agent, Robot) and agent.gripper_included:
                        agent_joint_angles[4]=0.01
                    #print('Action',agent_joint_angles)
            if isinstance(agent, Robot) and agent.action_duplication is not None:
                agent_joint_angles = np.concatenate([[a]*d for a, d in zip(agent_joint_angles, self.robot.action_duplication)])
                return self.control_robot(agent.all_controllable_joints, agent_joint_angles, agent.gains, agent.forces)
               
                # stretch testing is above
                if agent.gripper_included:
                    agent.perform_special_gripper_action(action_name, agent.left_gripper_indices)

        # if step_sim:
            # Update all agent positions
            # for _ in range(self.frame_skip):
                # p.stepSimulation(physicsClientId=self.id)
                # self.update_targets()
                # self.slow_time()


    def control_robot(self, indices, target_angles, gains, forces):
        if type(gains) in [int, float]:
            gains = [gains]*len(indices)
        if type(forces) in [int, float]:
            forces = [forces]*len(indices)
        print('SJCM',indices,target_angles)
        return target_angles


    def step(self, action):
 
        #action = np.array([1.0,0.0,0.0])
        action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)
        action *= self.robot.action_multiplier

        #print('Robot action ',action)
        #joint_sim_action = self.take_step_robot( action, gains=None, forces=None, action_multiplier=0.05, step_sim=True)
        self.take_step(action)
        #self.robot_action_list.append(joint_sim_action)
        obs = self._get_obs()
       
        self.robot_current_pose,base_orient = self.robot.get_pos_orient(self.robot.base)
        tool_pos,_ = self.robot.get_pos_orient(self.robot.left_tool_joint) 
        #current_joint_angles = env.robot.get_joint_angles(env.robot.left_arm_joint_indices)

        
        #target_joint_angles = env.robot.ik(env.robot.left_tool_joint, pf, base_orient, ik_indices=env.robot.left_arm_ik_indices, max_iterations=200)
        #env.robot.set_joint_angles(env.robot.left_arm_joint_indices, target_joint_angles)


        # reward_shake = -np.linalg.norm(self.old_tool_pos - tool_pos)
        # reward_base_movement = -np.linalg.norm(self.robot_current_pose-self.robot_old_pose)
        reward_distance = -np.linalg.norm(self.target_pos - tool_pos) #Penalize distances away from target
        # reward_base_distance = -np.linalg.norm(self.target_pos[:2] - self.robot_current_pose[:2])
        # reward_action = -np.linalg.norm(action) #Penalize actions

        base_pos = self.robot_current_pose
        pf = self.target_pos
        yaw_orientation = np.arctan2(2*(base_orient[3]*base_orient[2]+base_orient[0]*base_orient[1]),1-2*(base_orient[1]*base_orient[1]+base_orient[2]*base_orient[2]))
        base_pos_set = [pf[0]-0.0*cos(yaw_orientation)-0.4,pf[1]-0.0*sin(yaw_orientation)-0.1,base_pos[2]]
 
        reward_base_distance = -np.linalg.norm(base_pos_set - self.robot_current_pose)


        current_joint_angles = self.robot.get_joint_angles(self.robot.left_arm_joint_indices)
        target_joint_angles = self.robot.ik(self.robot.left_tool_joint, pf, base_orient, ik_indices=self.robot.left_arm_ik_indices, max_iterations=200)
        
        reward_target_angles = -np.linalg.norm(target_joint_angles - current_joint_angles)

        reward_time = -self.iteration

        #end_eff_pos, end_eff_orient = self.robot.get_pos_orient(self.robot.left_end_effector)
        #self.generate_line( end_eff_pos, end_eff_orient, 0.5)

        print('Tool differen Difference',reward_distance)
        if abs(reward_distance) < 0.08:
            self.task_success = 1
            print('Task success done!')
        else:
            self.task_success = 0

        #reward = self.config('base_distance_weight')*reward_base_distance + self.config('task_success_threshold')*self.task_success + self.config('shake_weight')*reward_shake + self.config('base_move_weight')*reward_base_movement +  self.config('distance_weight')*reward_distance + self.config('action_weight')*reward_action 
        reward = self.config('base_distance_weight')*reward_base_distance + self.config('task_success_threshold')*self.task_success  + self.config('tangle_weight')*reward_target_angles +  self.config('time_weight')*reward_time 


        self.total_reward = self.total_reward + reward
 
        # if self.gui:
        #     print('Task success:', self.task_success, 'Task total reward:', self.total_reward, ' Static reward', reward )

        #info = {'total_force_on_human': self.total_force_on_human, 'task_success': int(self.task_success >= self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        info = { 'task_success': self.task_success, 'action_robot_len': self.action_robot_len,  'obs_robot_len': self.obs_robot_len}
        

        self.robot_old_pose = self.robot_current_pose
        self.old_tool_pos = tool_pos
        #print('Observation space ', np.array(self.robot_obs_list).shape)
        if self.iteration>=200 or self.task_success==1:
            done = True
            print('Task success')
            #np.save('robot_observation_space_action_up.npy', np.array(self.robot_obs_list) )
            #np.save('robot_action_space_action_up.npy', np.array(self.robot_action_list) )
        else:
            done = False

        return obs, reward, done, info

   
    def generate_line(self, pos, orient, lineLen=0.5):
        
        p.removeAllUserDebugItems()
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
        
        #toY = [pos[0] - lineLen * dir1[0], pos[1] - lineLen * dir1[1], pos[2] - lineLen * dir1[2]]
        p.addUserDebugLine(pos, toX, [1, 0, 0], 5)
        p.addUserDebugLine(pos, toY, [0, 1, 0], 5)
        p.addUserDebugLine(pos, toZ, [0, 0, 1], 5)

        #p.addUserDebugLine(pos, to1, [0, 1, 1], 5, 3)
        #p.addUserDebugLine(pos, to2, [0, 1, 1], 5, 3)
        #p.addUserDebugLine(to2, to1, [0, 1, 1], 5, 3)

    def _get_obs(self, agent=None):

        target_pos_real_static = self.target_pos
        target_pos_real, target_orient_real = self.robot.convert_to_realworld(self.target_pos)

        base_pos, base_orient_quat = self.robot.get_base_pos_orient()
        end_eff_pos, end_eff_orient = self.robot.get_pos_orient(self.robot.left_end_effector, center_of_mass=False, convert_to_realworld=True)

        base_orient_euler = np.array(p.getEulerFromQuaternion(np.array(base_orient_quat), physicsClientId=self.id))

        pose_orient = np.array([base_pos[0],base_pos[1],base_orient_euler[2]])
        #print('Robot euler angles',int(base_orient_euler[2]/np.pi*180) )

        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)
        # Fix joint angles to be in [-pi, pi]
        robot_joint_angles = (np.array(robot_joint_angles) + np.pi) % (2*np.pi) - np.pi

        robot_obs = np.concatenate([target_pos_real, target_pos_real-end_eff_pos , end_eff_pos, end_eff_orient, robot_joint_angles]).ravel()
                                    # 3++                             3+++                3++       base_orient_euler     4++          3
        #print('Robot observations ', len(robot_obs))
        #print('Robot angles ',robot_obs)
 #        Robot angles  [ 3.45736027e-01 -2.26874888e-01  9.21452165e-01 -1.19052958e+00
 # -1.14407837e-02  1.15106165e-01  1.53626561e+00 -2.15434104e-01
 #  8.06345999e-01  5.32096834e-04 -4.79896989e-04 -2.83062388e-03
 #  9.99995708e-01  7.80223108e-01 -6.76220062e-07  7.37511972e-01]

        #print('Robot angles ', robot_joint_angles-self.init_robot_joint_angles  )
        self.robot_obs_list.append(robot_obs)

        self.init_robot_joint_angles = robot_joint_angles
        return robot_obs


    def reset(self):
        super(StretchTestingEnv, self).reset()
        #self.build_assistive_env()

        self.build_assistive_env('bed', fixed_human_base=False)

        self.furniture.set_friction(self.furniture.base, friction=5)

        # Setup human in the air and let them settle into a resting pose on the bed
        joints_positions = [(self.human.j_right_shoulder_x, 30)]
        self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=None)
        
        self.human.set_base_pos_orient([1.65, 0.15, 1.00], [-np.pi/2.0, 0, np.pi])
        self.furniture.set_base_pos_orient([1.85, 0.25, 0.00], [0, 0, -np.pi])

        self.human.set_gravity(0, 0, -10)

        target_ee_pos = np.array([1.25, 0.25, 0]) + self.np_random.uniform(-0.05, 0.05, size=3)
        target_ee_orient = self.get_quaternion([0, 0, 2*np.random.random()])
        #target_ee_orient = np.array([0, 0, 0, 1 ])
        # Not in itch scratch only here
        #self.robot.reset_joints()
        self.robot.skip_pose_optimization = True
        #self.init_robot_pose(target_ee_pos, target_ee_orient, [None], [None], arm='left', tools=[], collision_objects=[],wheelchair_enabled=True)
        
        self.robot.randomize_init_joint_angles(self.task)
        self.robot.set_base_pos_orient(target_ee_pos, target_ee_orient)

        #self.init_robot_pose(target_ee_pos, target_ee_orient, start_pos_orient, target_pos_orients, arm='right', tools=[], collision_objects=[], wheelchair_enabled=True, right_side=True, max_iterations=3)
        #self.generate_target()
        base_pos, base_orient = self.robot.get_pos_orient(self.robot.base)
        pf = self.generate_target_point(base_pos,base_orient)
        self.generate_target()
        
        #pf = env.target_pos
        yaw_orientation = np.arctan2(2*(base_orient[3]*base_orient[2]+base_orient[0]*base_orient[1]),1-2*(base_orient[1]*base_orient[1]+base_orient[2]*base_orient[2]))
        base_pos_set = [pf[0]-0.0*cos(yaw_orientation)-0.4,pf[1]-0.0*sin(yaw_orientation)-0.1,base_pos[2]]
    
        self.robot.set_base_pos_orient( base_pos_set, base_orient)

        target_pos = pf
        self.target = self.create_sphere(radius=0.02, mass=0.0, pos=target_pos, visual=True, collision=False, rgba=[0, 1, 1, 1])
        self.target_pos = np.array(target_pos)
        self.target.set_base_pos_orient(self.target_pos, [0, 0, 0, 1])

        #p.setGravity(0, 0, -9.81, physicsClientId=self.id)  #changes the whole simulation response
        #if not self.robot.mobile:
        self.robot.set_gravity(0, 0, -9.81)
        self.init_robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)
        print('Robot pose',self.robot.get_pos_orient(self.robot.base) , 'Target pose', self.target_pos)
        print('Init joint angles', self.init_robot_joint_angles)
        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        self.init_env_variables()
        self.total_reward = 0
        return self._get_obs()

    
    def generate_target(self):

        self.total_reward = 0
        self.robot_old_pose,_ = self.robot.get_pos_orient(self.robot.base)
        self.old_tool_pos,_ = self.robot.get_pos_orient(self.robot.left_end_effector)


        target_pos = [ 0.5+np.random.random(), 0.5+np.random.random(), 0.7+(np.random.random()/10)]
        self.target = self.create_sphere(radius=0.02, mass=0.0, pos=target_pos, visual=True, collision=False, rgba=[0, 1, 1, 1])
        self.target_pos = np.array(target_pos)
        self.target.set_base_pos_orient(self.target_pos, [0, 0, 0, 1])

        #target_pos, target_orient = p.multiplyTransforms(arm_pos, arm_orient, self.target_on_arm, [0, 0, 0, 1], physicsClientId=self.id)

        #self.create_sphere(radius=0.1, mass=0.0, pos=arm_pos, visual=True, collision=False, rgba=[0, 0, 1, 1])

    def generate_target_point(self, pos, orient):
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


    
    def generate_target_human(self):

        self.total_reward=0
        
        self.robot_old_pose,_orient_ = self.robot.get_pos_orient(self.robot.base)
        self.robot_old_arm,_orient_ = self.robot.get_pos_orient(self.robot.left_end_effector)
        # Randomly select either upper arm or forearm for the target limb to scratch
        # if self.human.gender == 'male':
        #     self.limb, length, radius = [[self.human.right_shoulder, 0.279, 0.043], [self.human.right_elbow, 0.157, 0.033], [self.human.left_shoulder, 0.279, 0.043], [self.human.left_elbow, 0.157, 0.033], [self.human.right_hip, 0.289, 0.078], [self.human.right_knee, 0.172, 0.063], [self.human.left_hip, 0.289, 0.078], [self.human.left_knee, 0.172, 0.063] ][self.np_random.randint(8)]
        # else:
        #     self.limb, length, radius = [[self.human.right_shoulder, 0.264, 0.0355], [self.human.right_elbow, 0.134, 0.027], [self.human.left_shoulder, 0.264, 0.0355], [self.human.left_elbow, 0.134, 0.027], [self.human.right_hip, 0.279, 0.0695], [self.human.right_knee, 0.164, 0.053], [self.human.left_hip, 0.279, 0.0695], [self.human.left_knee,  0.164, 0.053] ][self.np_random.randint(8)]

        if self.human.gender == 'male':
            self.limb, length, radius = [[self.human.right_shoulder, 0.279, 0.043], [self.human.right_elbow, 0.157, 0.033], [self.human.right_hip, 0.289, 0.078], [self.human.right_knee, 0.172, 0.063] ][self.np_random.randint(4)]
        else:
            self.limb, length, radius = [[self.human.right_shoulder, 0.264, 0.0355], [self.human.right_elbow, 0.134, 0.027], [self.human.right_hip, 0.279, 0.0695], [self.human.right_knee, 0.164, 0.053] ][self.np_random.randint(4)]
           
        self.target_on_arm = self.util.point_on_capsule(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -length*2]), radius=radius, theta_range=(0, np.pi*2))
        arm_pos, arm_orient = self.human.get_pos_orient(self.limb)
        target_pos, target_orient = p.multiplyTransforms(arm_pos, arm_orient, self.target_on_arm, [0, 0, 0, 1], physicsClientId=self.id)

        self.target = self.create_sphere(radius=0.02, mass=0.0, pos=target_pos, visual=True, collision=False, rgba=[0, 1, 1, 1])
        #self.create_sphere(radius=0.1, mass=0.0, pos=arm_pos, visual=True, collision=False, rgba=[0, 0, 1, 1])

        self.update_targets_human()



    def update_targets_human(self):
        arm_pos, arm_orient = self.human.get_pos_orient(self.limb)
        target_pos, target_orient = p.multiplyTransforms(arm_pos, arm_orient, self.target_on_arm, [0, 0, 0, 1], physicsClientId=self.id)
        self.target_pos = np.array(target_pos)
        self.target.set_base_pos_orient(self.target_pos, [0, 0, 0, 1])


