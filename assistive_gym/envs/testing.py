from turtle import position
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from .env import AssistiveEnv
from .agents import furniture
from .agents.furniture import Furniture

target = 0.9

class TestingEnv(AssistiveEnv):
    def __init__(self, robot):
        super(TestingEnv, self).__init__(robot=robot, task='testing', obs_robot_len=(18 + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices) if robot.mobile else 0)))

    #TODO start to use math of trapezoidal trajectories

    def step(self, action):
        done = self.iteration >= 100
        reward = None
        info = None

        self.take_step(action)

        obs = self._get_obs()

        return obs, reward, done, info

    def _get_obs(self, agent=None):
        lift_angle = self.robot.get_joint_angles([3])

        lift_velocity = self.robot.get_velocity(3)
        lift_velocity = np.linalg.norm(lift_velocity)
        
        lift_force = self.robot.get_force_torque_sensor(3)
        lift_force = np.linalg.norm(lift_force)

        self.write_graphs(lift_angle,lift_velocity,lift_force)

        return lift_angle

    def write_graphs(self,lift_angle,lift_velocity,lift_force):
        time = self.iteration

        self.times = np.append(self.times,time)
        self.lift_angles = np.append(self.lift_angles,lift_angle)
        self.lift_velocities = np.append(self.lift_velocities,lift_velocity)
        self.lift_forces = np.append(self.lift_forces,lift_force)

        if self.iteration >= 100:
            plt.subplot(311)
            plt.plot(self.times,self.lift_angles)
            plt.axis([0,self.iteration,0,1])

            plt.subplot(312)
            plt.plot(self.times,self.lift_velocities)
            plt.axis([0,self.iteration,-.2,.2])

            plt.subplot(313)
            plt.plot(self.times,self.lift_forces)
            plt.axis([0,self.iteration,-.1,.1])
            plt.show()

    def reset(self):
        super(TestingEnv, self).reset()

        self.times = np.array([])
        self.lift_angles = np.array([])
        self.lift_velocities = np.array([])
        self.lift_forces = np.array([])

        self.build_assistive_env()
        # self.generate_target()
        self.init_robot_pose([],[],[],[])
        self.init_env_variables()

        p.resetDebugVisualizerCamera(cameraDistance=1.10, cameraYaw=40, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.75], physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        return self._get_obs()

    # def generate_target(self):
    #     self.target_pos = [0, 0, 1.2]
    #     # self.target_pos = self.target_ee_pos + [0, 0, -1]
    #     self.target = self.create_sphere(radius=0.01, mass=0.0, pos=self.target_pos, collision=False, rgba=[0, 1, 0, 1])
    #     self.target.set_base_pos_orient(self.target_pos, [0, 0, 0, 1])

##### urdf file with lift info #####

#   <link name="base_link">
#     <inertial>
#       <origin rpy="0 0 0" xyz="-0.109461304328163 -0.000741018909047708 0.0914915269429946"/>
#       <mass value="1.0723782659782"/>
#       <inertia ixx="0.00310580907710135" ixy="1.5182848191076E-06" ixz="0.00041690466732394" iyy="0.00433798719991832" iyz="1.33487716258445E-05" izz="0.0037204727467362"/>
#     </inertial>

#   <joint name="joint_mast" type="fixed">
#     <origin rpy="1.5707963267949 0 0.00925318926571245" xyz="-0.07 0.134999999999998 0.0284000000000001"/>
#     <parent link="base_link"/>
#     <child link="link_mast"/>
#     <axis xyz="0 0 0"/>
#   </joint>

#   <link name="link_mast">
#     <inertial>
#       <origin rpy="0 0 0" xyz="0.00755818572975822 0.773971284176834 0.00647313086620024"/>
#       <mass value="0.749143203376401"/>
#       <inertia ixx="0.0709854511954588" ixy="-0.00433428742758457" ixz="-0.000186110788697573" iyy="0.000437922053342648" iyz="-0.00288788257713431" izz="0.071104808501661"/>
#     </inertial>

#   <joint name="joint_lift" type="prismatic">
#     <origin rpy="-1.57079632679552 1.5615431375292 -6.2942004366467E-13" xyz="-0.0369217062323472 0.165471199999996 -0.000341653286793524"/>
#     <parent link="link_mast"/>
#     <child link="link_lift"/>
#     <axis xyz="0 0 1"/>
#     <limit effort="100" lower="0.0" upper="1.1" velocity="1.0"/>
#   </joint>

#####