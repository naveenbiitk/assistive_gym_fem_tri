import numpy as np
import pybullet as p
import optparse
from scipy.spatial.transform import Rotation as R
import os
import cv2

from .env import AssistiveEnv
from .agents import furniture
from .agents.furniture import Furniture



class ObjectHandoverEnv(AssistiveEnv):

    def __init__(self, robot, human):
        super(ObjectHandoverEnv, self).__init__(robot=robot, human=human, task='object_handover', obs_robot_len=(23 + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices) if robot.mobile else 0)), obs_human_len=(24 + len(human.controllable_joint_indices)))
        self.general_model = True
        # Parameters for personalized human participants
        self.gender = 'female'
        self.body_shape_filename = '%s_1.pkl' % self.gender
        self.human_height = 1.6


    def step(self, action):

        self.distance_threshold = 0.1/1.5

        if self.human.controllable:
            action = np.concatenate([action['robot'], action['human']])
        self.take_step(action)

        obs = self._get_obs()
        # print(np.array_str(obs, precision=3, suppress_small=True))

        # Get human preferences
        end_effector_velocity = np.linalg.norm(self.robot.get_velocity(self.robot.left_end_effector))
        preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity, total_force_on_human=self.total_force_on_human, tool_force_at_target=self.tool_force_at_target)

        tool_pos = self.tool.get_pos_orient(1)[0]
        reward_distance = -np.linalg.norm(self.target_pos - tool_pos) # Penalize distances away from target
        reward_action = -np.linalg.norm(action) # Penalize actions
        reward_force_scratch = 0.0 # Reward force near the target
        #if self.target_contact_pos is not None and np.linalg.norm(self.target_contact_pos - self.prev_target_contact_pos) > 0.001 and self.tool_force_at_target < 10:
        # Encourage the robot to move around near the target to simulate scratching
        

        # Discourage movement only for stretch not for pr2
        self.robot_current_pose,_orient_ = self.robot.get_pos_orient(self.robot.base)
        reward_movement = -3*np.linalg.norm(self.robot_current_pose-self.robot_old_pose)
        
        #only for pr2 not stretch
        #reward_movement=0

        self.robot_old_pose = self.robot_current_pose

        self.robot_current_arm,_orient_ = self.robot.get_pos_orient(self.robot.left_end_effector)
        
        reward_shake = 0
        if self.task_success>1:
            reward_shake = -1*np.linalg.norm(self.robot_current_arm-self.robot_old_arm)
        
        self.robot_old_arm = self.robot_current_arm
            

        #print(self.robot_current_pose)
        if abs(reward_distance)<self.distance_threshold: #0.03
            reward_force_scratch = 5
            self.task_success = 1

        if abs(reward_distance)<self.distance_threshold and self.tool_force_at_target<10: #0.03
            reward_force_scratch = 5
            self.task_success =+ 2

        # if self.total_force_on_human>10:
        #     print('Force on human: ',self.total_force_on_human)


        reward = reward_shake + reward_movement +  self.config('distance_weight')*reward_distance + self.config('action_weight')*reward_action + self.config('scratch_reward_weight')*reward_force_scratch + preferences_score

        self.total_reward=self.total_reward+reward
        self.prev_target_contact_pos = self.target_contact_pos

        if self.gui and self.tool_force_at_target > 0:
            print('Task success:', self.task_success, 'Tool force at target:', self.tool_force_at_target, reward_force_scratch)
            print('Task reward:', self.total_reward)

        #info = {'total_force_on_human': self.total_force_on_human, 'task_success': int(self.task_success >= self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        info = {'total_force_on_human': self.total_force_on_human, 'task_success': self.task_success, 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = self.iteration >= 200

        jt = self.robot.get_joint_angles(indices=self.robot.left_arm_joint_indices)
        #print('robot joint angles',jt)

        if not self.human.controllable:
            return obs, reward, done, info
        else:
            # Co-optimization with both human and robot controllable
            return obs, {'robot': reward, 'human': reward}, {'robot': done, 'human': done, '__all__': done}, {'robot': info, 'human': info}



    def get_total_force(self):
        total_force_on_human = np.sum(self.robot.get_contact_points(self.human)[-1])
        tool_force = np.sum(self.tool.get_contact_points()[-1])
        tool_force_at_target = 0
        target_contact_pos = None
        for linkA, linkB, posA, posB, force in zip(*self.tool.get_contact_points(self.human)):
            total_force_on_human += force
            # Enforce that contact is close to the target location
            if linkA in [0, 1] and np.linalg.norm(posB - self.target_pos) < self.distance_threshold:
                tool_force_at_target += force
                target_contact_pos = posB
        return total_force_on_human, tool_force, tool_force_at_target, None if target_contact_pos is None else np.array(target_contact_pos)



    def _get_obs(self, agent=None):
        tool_pos, tool_orient = self.tool.get_pos_orient(1)
        tool_pos_real, tool_orient_real = self.robot.convert_to_realworld(tool_pos, tool_orient)
        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)
        # Fix joint angles to be in [-pi, pi]
        robot_joint_angles = (np.array(robot_joint_angles) + np.pi) % (2*np.pi) - np.pi
        if self.robot.mobile:
            # Don't include joint angles for the wheels
            robot_joint_angles = robot_joint_angles[len(self.robot.wheel_joint_indices):]
        

        shoulder_pos = self.human.get_pos_orient(self.human.right_shoulder)[0]#5upperarm   agym_jt[6,:] 
        elbow_pos = self.human.get_pos_orient(self.human.right_elbow)[0]#7forearm          agym_jt[8,:]
        wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]#9j_right_wrist_x  agym_jt[10,:]  

        shoulder_pos_real, _ = self.robot.convert_to_realworld(shoulder_pos)
        elbow_pos_real, _ = self.robot.convert_to_realworld(elbow_pos)
        wrist_pos_real, _ = self.robot.convert_to_realworld(wrist_pos)
        target_pos_real, _ = self.robot.convert_to_realworld(self.target_pos)

        #human_pose_predicted = self.call_bp_cb()
        #print("Human pose predicted", shoulder_pos- human_pose_predicted[6] )

        self.total_force_on_human, self.tool_force, self.tool_force_at_target, self.target_contact_pos = self.get_total_force()
        robot_obs = np.concatenate([tool_pos_real, tool_orient_real, tool_pos_real - target_pos_real, target_pos_real, robot_joint_angles, shoulder_pos_real, elbow_pos_real, wrist_pos_real, [self.tool_force]]).ravel()
        if agent == 'robot':
            return robot_obs
        if self.human.controllable:
            human_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
            tool_pos_human, tool_orient_human = self.human.convert_to_realworld(tool_pos, tool_orient)
            shoulder_pos_human, _ = self.human.convert_to_realworld(shoulder_pos)
            elbow_pos_human, _ = self.human.convert_to_realworld(elbow_pos)
            wrist_pos_human, _ = self.human.convert_to_realworld(wrist_pos)
            target_pos_human, _ = self.human.convert_to_realworld(self.target_pos)
            human_obs = np.concatenate([tool_pos_human, tool_orient_human, tool_pos_human - target_pos_human, target_pos_human, human_joint_angles, shoulder_pos_human, elbow_pos_human, wrist_pos_human, [self.total_force_on_human, self.tool_force_at_target]]).ravel()
            if agent == 'human':
                return human_obs
            # Co-optimization with both human and robot controllable
            return {'robot': robot_obs, 'human': human_obs}
        return robot_obs




    def reset(self):
        super(ObjectHandoverEnv, self).reset()

        self.build_assistive_env('wheelchair')
        self.furniture.set_on_ground()

        if self.robot.wheelchair_mounted:
            wheelchair_pos, wheelchair_orient = self.furniture.get_base_pos_orient()
            self.robot.set_base_pos_orient(wheelchair_pos + np.array(self.robot.toc_base_pos_offset[self.task]), [0, 0, -np.pi/2.0])


        if self.general_model:
            # Randomize the human body shape
            gender = self.np_random.choice(['male', 'female'])
            # body_shape = self.np_random.randn(1, self.human.num_body_shape)
            body_shape = self.np_random.uniform(-2, 5, (1, self.human.num_body_shape))
            # human_height = self.np_random.uniform(1.59, 1.91) if gender == 'male' else self.np_random.uniform(1.47, 1.78)
            human_height = self.np_random.uniform(1.5, 1.9)
        else:
            gender = self.gender
            body_shape = self.body_shape_filename
            human_height = self.human_height


        # Update robot and human motor gains
        self.robot.motor_gains = self.human.motor_gains = 0.005


        joint_angles = [(self.human.j_left_hip_x, -90), (self.human.j_right_hip_x, -90), (self.human.j_left_knee_x, 70), (self.human.j_right_knee_x, 70), (self.human.j_left_shoulder_z, -45), (self.human.j_right_shoulder_z, 45), (self.human.j_left_elbow_y, -90), (self.human.j_right_elbow_y, 90)]
        # u = self.np_random.uniform
        # joint_angles += [(self.human.j_waist_x, u(-30, 45)), (self.human.j_waist_y, u(-45, 45)), (self.human.j_waist_z, u(-30, 30)), (self.human.j_lower_neck_x, u(-30, 30)), (self.human.j_lower_neck_y, u(-30, 30)), (self.human.j_lower_neck_z, u(-10, 10)), (self.human.j_upper_neck_x, u(-45, 45)), (self.human.j_upper_neck_y, u(-30, 30)), (self.human.j_upper_neck_z, u(-30, 30))]
        # joint_angles += [(self.human.j_waist_x, u(-20, 30)), (self.human.j_waist_y, u(-45, 0)), (self.human.j_waist_z, u(0, 30)), (self.human.j_lower_neck_x, u(-30, 30)), (self.human.j_lower_neck_y, u(-30, 30)), (self.human.j_lower_neck_z, u(-10, 10)), (self.human.j_upper_neck_x, u(-30, 30)), (self.human.j_upper_neck_y, u(-30, 30)), (self.human.j_upper_neck_z, u(-30, 30))]
        joint_angles += [(j, self.np_random.uniform(-10, 10)) for j in (self.human.j_waist_x, self.human.j_waist_y, self.human.j_waist_z, self.human.j_lower_neck_x, self.human.j_lower_neck_y, self.human.j_lower_neck_z, self.human.j_upper_neck_x, self.human.j_upper_neck_y, self.human.j_upper_neck_z)]
        self.human.init(self.directory, self.id, self.np_random, gender=gender, height=human_height, body_shape=body_shape, joint_angles=joint_angles, position=[0, 0, 0], orientation=[0, 0, 0])



        # Place human in chair
        chair_seat_position = np.array([0, 0.05, 0.6])
        self.human.set_base_pos_orient(self.furniture.get_base_pos_orient()[0] + chair_seat_position - self.human.get_vertex_positions(self.human.bottom_index), [0, 0, 0, 1])



        joints_positions = [(self.human.j_right_elbow, -90), (self.human.j_left_elbow, -90), (self.human.j_right_hip_x, -90), (self.human.j_right_knee, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee, 80)]
        joints_positions += [(self.human.j_head_x, self.np_random.uniform(-30, 30)), (self.human.j_head_y, self.np_random.uniform(-30, 30)), (self.human.j_head_z, self.np_random.uniform(-30, 30))]
        self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None)

        self.generate_target()

        p.resetDebugVisualizerCamera(cameraDistance=1.10, cameraYaw=55, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.75], physicsClientId=self.id)

        self.tool.init(self.robot, self.task, self.directory, self.id, self.np_random, right=True, mesh_scale=[0.045]*3, alpha=0.75)

        target_ee_pos = np.array([-0.2, -0.5, 1.1]) + self.np_random.uniform(-0.05, 0.05, size=3)
        target_ee_orient = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task])
        self.init_robot_pose(target_ee_pos, target_ee_orient, [(target_ee_pos, target_ee_orient), (self.target_pos, None)], [(self.target_pos, target_ee_orient)], arm='right', tools=[self.tool], collision_objects=[self.human, self.furniture])

        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.right_gripper_indices, self.robot.gripper_pos[self.task], set_instantly=True)

        if not self.robot.mobile:
            self.robot.set_gravity(0, 0, 0)
        self.human.set_gravity(0, 0, 0)
        self.tool.set_gravity(0, 0, 0)


        p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=10, physicsClientId=self.id)


                # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        self.init_env_variables()
        return self._get_obs()



    
    def generate_target(self):

        self.total_reward=0
        
        self.robot_old_pose,_orient_ = self.robot.get_pos_orient(self.robot.base)
        self.robot_old_arm,_orient_ = self.robot.get_pos_orient(self.robot.left_end_effector)

        if self.human.gender == 'male':
            self.limb, length, radius = self.human.right_elbow, 0.157, 0.033
        else:
            self.limb, length, radius = self.human.right_elbow, 0.134, 0.027
           
        self.target_on_arm = self.util.point_on_capsule(p1=np.array([0, 0, -2*length]), p2=np.array([0, 0, -length*2.5]), radius=radius, theta_range=(0, np.pi*2))
        arm_pos, arm_orient = self.human.get_pos_orient(self.limb)
        target_pos, target_orient = p.multiplyTransforms(arm_pos, arm_orient, self.target_on_arm, [0, 0, 0, 1], physicsClientId=self.id)

        self.target = self.create_sphere(radius=0.02, mass=0.0, pos=target_pos, visual=True, collision=False, rgba=[0, 1, 1, 1])
        #self.create_sphere(radius=0.1, mass=0.0, pos=arm_pos, visual=True, collision=False, rgba=[0, 0, 1, 1])

        self.update_targets()



    def update_targets(self):
        arm_pos, arm_orient = self.human.get_pos_orient(self.limb)
        target_pos, target_orient = p.multiplyTransforms(arm_pos, arm_orient, self.target_on_arm, [0, 0, 0, 1], physicsClientId=self.id)
        self.target_pos = np.array(target_pos)
        self.target.set_base_pos_orient(self.target_pos, [0, 0, 0, 1])


    
    def get_depth_image(self):
        # * take image after reward computed
        img, depth = self.get_camera_image_depth()

        far = 2.406
        near = 1.406

        a_ = (far-near)/(np.max(depth) - np.min(depth))
        b_ = - (far*np.min(depth)-near*np.max(depth))/(np.max(depth) - np.min(depth))
        depth = depth*(a_)+b_

        depth = depth*1000

        depth = depth[50:178, 73:127]

        #filename='/nethome/nnagarathinam6/Documents/joint_reaching_evaluation/'
        #outfile = filename + "after_depth" + str(1) + ".npy"
        #np.save(outfile, depth)
        
        #file = filename + "depth" + str(1) + ".png"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('color', img)
        cv2.waitKey(0) 

        return depth

