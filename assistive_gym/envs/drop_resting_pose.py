import numpy as np
import pybullet as p
from .env import AssistiveEnv

# matt's stuff
from ..human_params import HumanJoints
from .lamsey_utils.chair_builder import build_chair
from .lamsey_utils.coordinate_frame import origin_coordinate_system
# from .lamsey_utils.test_smplx_poses import armchair_seated, test_pose
from .lamsey_utils.smpl_dict import SMPLDict
from .lamsey_utils.SMPLXProcessor import load_pose_from_pkl
from .lamsey_utils.test_frames import Frames

smpl_dict = SMPLDict()
from pytorch3d import transforms
from torch import tensor

from math import pi
import os

from .agents.human import Human
from .agents.human import right_arm_joints, left_arm_joints, torso_joints, head_joints, all_joints
controllable_joints = right_arm_joints + left_arm_joints + torso_joints + head_joints

l_hand_index = HumanJoints.left_wrist.value
r_hand_index = HumanJoints.right_wrist.value
head_index = HumanJoints.head.value


def configure_human(human):
    human.impairment = None
    human.set_all_joints_stiffness(0.02)
    human.set_whole_body_frictions(lateral_friction=50., spinning_friction=10., rolling_friction=10.)

    # joint_pos = default_sitting_pose(human)
    # pose = armchair_seated()
    # pose = test_pose()
    frames = Frames().armchair_frames
    frame = "armchair001_stageII.pkl"
    pkl_path = os.path.expanduser("~") + "hrl_git/assistive_gym_fem_tri/assistive_gym/envs/lamsey_utils/SAMP/pkl/" + frame
    _, _, pose, _, _ = load_pose_from_pkl(pkl_path, frames[frame])
    pose = pose.reshape(165, 1)
    joint_pos = unpack_smplx_pose(human, pose)
    human.setup_joints(joint_pos, use_static_joints=False, reactive_force=None)

    start_pos = [0., 0., 0.85]
    # start_pos = [0, 0, 1.2]
    # start_orient = [0, 0, 0, 1]

    start_angles = [-pi/2., 0., pi]  # XYZ Euler Angles
    start_orient = transforms.matrix_to_quaternion(transforms.euler_angles_to_matrix(tensor(start_angles), 'XYZ')).numpy()

    # start_orient = [-0.0898752, 0, 0.7058752, 0.7026113]

    # commented: pure x axis rotation by -0.2rad to -0.5rad
    # start_orient = [ -0.0998334, 0, 0, 0.9950042 ]
    # start_orient = [ -0.1494381, 0, 0, 0.9887711 ]
    # start_orient = [ -0.1986693, 0, 0, 0.9800666 ]
    # start_orient = [ -0.247404, 0, 0, 0.9689124 ]

    human.set_base_pos_orient(start_pos, start_orient)
    # human.set_on_ground()

    joint_i = [pose[0] for pose in joint_pos]
    joint_th = [pose[1] for pose in joint_pos]
    joint_gains = [0.] * len(joint_i)
    # forces = [50.] * len(joint_i)
    forces = [0.] * len(joint_i)

    # tweak joint control
    for i in range(len(joint_gains)):
        if i not in controllable_joints or i in right_arm_joints:
            joint_gains[i] = 0.
            forces[i] = 0.

    human.control(joint_i, joint_th, joint_gains, forces)
    human.is_controllable = True

# def set_joint_stiffnesses(human):
#     human.set_joint_stiffness(human.j_)


def default_sitting_pose(human):
    # Arms
    joint_pos = [(human.j_right_shoulder_x, 30.),
                 (human.j_left_shoulder_x, -30.),
                 (human.j_right_shoulder_y, 0.),
                 (human.j_left_shoulder_y, 0.),
                 (human.j_right_elbow, -90.),
                 (human.j_left_elbow, -90.)]

    # Legs
    joint_pos += [(human.j_right_knee, 90.),
                  (human.j_left_knee, 90.),
                  (human.j_right_hip_x, -90.),
                  (human.j_left_hip_x, -90.)]

    # Torso
    joint_pos += [(human.j_waist_x, 0.)]
    return joint_pos


def unpack_smplx_pose(human, pose):
    # unpack smplx pose
    right_shoulder = pose[smpl_dict.get_pose_ids("right_shoulder")]
    left_shoulder = pose[smpl_dict.get_pose_ids("left_shoulder")]
    right_elbow = pose[smpl_dict.get_pose_ids("right_elbow")]
    left_elbow = pose[smpl_dict.get_pose_ids("left_elbow")]
    right_wrist = pose[smpl_dict.get_pose_ids("right_wrist")]
    left_wrist = pose[smpl_dict.get_pose_ids("left_wrist")]
    right_hip = pose[smpl_dict.get_pose_ids("right_hip")]
    left_hip = pose[smpl_dict.get_pose_ids("left_hip")]
    right_knee = pose[smpl_dict.get_pose_ids("right_knee")]
    left_knee = pose[smpl_dict.get_pose_ids("left_knee")]
    lower_spine = pose[smpl_dict.get_pose_ids("lower_spine")]
    right_collar = pose[smpl_dict.get_pose_ids("right_collar")]
    left_collar = pose[smpl_dict.get_pose_ids("left_collar")]
    neck = pose[smpl_dict.get_pose_ids("neck")]
    head = pose[smpl_dict.get_pose_ids("head")]

    # Arms
    joint_pos = [(human.j_right_shoulder_x, -right_shoulder[2] + pi/2.),
                 (human.j_right_shoulder_y, -right_shoulder[1]),
                 (human.j_right_shoulder_z, right_shoulder[0]),
                 (human.j_left_shoulder_x, -left_shoulder[2] - pi/2.),
                 (human.j_left_shoulder_y, left_shoulder[1]),
                 (human.j_left_shoulder_z, -left_shoulder[0]),
                 (human.j_right_elbow, -right_elbow[1]),
                 (human.j_left_elbow, left_elbow[1])]
                 # (human.j_right_wrist_x, right_wrist[0]),
                 # (human.j_left_wrist_x, left_wrist[0])]

    # Legs
    joint_pos += [(human.j_left_hip_x, left_hip[0]),
                  (human.j_left_hip_y, -left_hip[2]),
                  (human.j_left_hip_z, left_hip[1]),
                  (human.j_right_hip_x, right_hip[0]),
                  (human.j_right_hip_y, -right_hip[2]),
                  (human.j_right_hip_z, right_hip[1]),
                  (human.j_left_knee, left_knee[0]),
                  (human.j_right_knee, right_knee[0])]

    # Torso
    joint_pos += [(human.j_waist_x, -lower_spine[0]),
                  (human.j_waist_y, lower_spine[1]),
                  (human.j_waist_z, lower_spine[2]),]
                  # (human.j_right_pecs_x, -right_collar[2]),
                  # (human.j_right_pecs_y, -right_collar[1]),
                  # (human.j_right_pecs_z, right_collar[0]),
    #               (human.j_left_pecs_x, left_collar[0]),
    #               (human.j_left_pecs_y, left_collar[1]),
    #               (human.j_left_pecs_z, left_collar[2])]

    # Head
    joint_pos += [(human.j_neck, neck[0]),
                  (human.j_head_x, head[0]),
                  (human.j_head_y, -head[1]),
                  (human.j_head_z, -head[2])]

    for i in range(len(joint_pos)):
        joint = joint_pos[i]
        joint_pos[i] = (joint[0], np.rad2deg(joint[1]))

    return joint_pos


class BasePoseEnv(AssistiveEnv):
    def __init__(self, human):
        super(BasePoseEnv, self).__init__(robot=None, human = human, task='pose_analysis', )

    def step(self, action):
        self.take_step(action, action_multiplier=1.)

        observation = self._get_obs()

        if observation["human"]:
            if self.once:
                self.human.set_on_ground()
                for j in all_joints:
                    # damping coefficients
                    a_d = 20.
                    l_d = 1.
                    p.changeDynamics(self.human.body, j, physicsClientId=self.id, angularDamping=a_d, linearDamping=l_d)
                p.setGravity(0., 0., -9.81)
                # p.setGravity(0., 0., -9.81/2.)
                self.human_pose = self.human.get_joint_angles()
                self.once = False
                # input()

        self.steps += 1
        reward = 0
        done = False if self.steps < self.max_steps else True
        # done = True if observation["human"] else False
        # if self.steps == 1:
        #     input("Episode Initialized. Press Enter to Continue.")
        info = {"n/a": 'n/a'}  # must be a dict

        return observation, reward, done, info

    def _get_obs(self, agent=None):
        # observation = self.human.get_joint_angles()
        # observation = self.human.get_pos_orient(self.human.head)
        observation = {"human": (), "chair": (), "human_pose": ()}

        if self.human is not None:
            # get pose
            observation.update({"human_pose": self.human.get_joint_angles()})

            # get contacts
            human_contacts = p.getContactPoints(self.human.body)
            if human_contacts:
                for contact in human_contacts:
                    linkIndexA = contact[3]
                    linkIndexB = contact[4]

                    if linkIndexA >= 0 and linkIndexB >= 0:
                        if HumanJoints(linkIndexA).name is "head" or HumanJoints(linkIndexB).name is "head":
                            observation.update({"human": human_contacts})
                            # self.plot_contact_points([contact], rgba=[0., 0.75, 0., 0.75])
                            # input()

        if self.plane_chair is not None:
            chair_contact = ()
            for component in self.plane_chair:
                chair_contact += p.getContactPoints(component)

            observation.update({"chair": chair_contact})
            if self.steps == self.max_steps - 1:
                self.plot_contact_points(observation["chair"])
                # input("Episode Complete. Press Enter to Continue.")

        return observation

    def reset(self):
        super(BasePoseEnv, self).reset()
        self.build_assistive_env(fixed_human_base=False)
        # plane_path = os.path.join(self.directory, "primitives", "plane_chair.urdf")
        # self.plane_chair = p.loadURDF(plane_path, flags=p.URDF_USE_SELF_COLLISION, physicsClientId=self.id)
        base_height = 0.375  # seat height
        self.plane_chair = build_chair(arm_separation=0.65,
                                       left_arm_height=base_height + 0.35,
                                       right_arm_height=base_height + 0.35,
                                       seat_height=base_height,
                                       arm_length=0.4)
        origin_coordinate_system()

        # Human init
        configure_human(self.human)
        # print(self.human.__dict__.keys())
        right_hand_pos, right_hand_orient = self.human.get_pos_orient(link=self.human.right_wrist)
        head_pos, head_orient = self.human.get_pos_orient(link=self.human.head)

        # print("right hand to head distance:")
        # print(head_pos - right_hand_pos)
        # p.addUserDebugLine(right_hand_pos, head_pos, [1, 0, 0], 2)

        # Right Arm IK
        # ik_solution = np.rad2deg(self.human.ik(target_joint=self.human.right_wrist, target_pos=head_pos, target_orient=right_hand_orient, ik_indices=self.human.right_arm_joints, max_iterations=200))
        # control_forces = [500.] * len(ik_solution)
        # control_gains = [500.] * len(ik_solution)
        # self.human.control(indices=self.human.right_arm_joints, target_angles=ik_solution, gains=control_gains, forces=control_forces)
        # target_pose = []
        # for i in range(len(self.human.right_arm_joints)):
        #     target_pose.append((self.human.right_arm_joints[i], np.rad2deg(ik_solution[i])))
        #
        # print(ik_solution)
        # self.human.setup_joints(target_pose)

        self.human.set_on_ground()

        p.setGravity(0, 0, 0)

        p.resetDebugVisualizerCamera(cameraDistance=1.10, cameraYaw=40, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.75], physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
        return self._get_obs()

    # Extra  stuff
    def plot_contact_points(self, contact_points, rgba=[1, 0, 0, 0.75]):
        point_locations = []
        target_joints = [l_hand_index, r_hand_index, head_index]
        print(target_joints)

        for point in contact_points:
            contactFlag = point[0]
            bodyUniqueIdA = point[1]
            bodyUniqueIdB = point[2]
            linkIndexA = point[3]
            linkIndexB = point[4]
            positionOnA = point[5]
            positionOnB = point[6]
            contactNormalOnB = point[7]
            contactDistance = point[8]
            normalForce = point[9]

            if linkIndexB in target_joints:
                print("Target Link Contact at: " + str(positionOnB))
                print(HumanJoints(linkIndexB).name)

            # B is the limb, A is the chair

            # p.addUserDebugLine(lineFromXYZ=positionOnB,
            #                    lineToXYZ=(0., 0., 0.),
            #                    lineColorRGB=(1., 0., 0.),
            #                    lineWidth=10.,
            #                    lifeTime=1.)

            # plot_obj = p.createVisualShape(shapeType=p.GEOM_SPHERE,
            #                                radius=0.1,
            #                                visualFramePosition=positionOnB)

            # p.createMultiBody(baseVisualShapeIndex=plot_obj)

            point_locations.append(positionOnB)

        self.create_spheres(radius=0.05, batch_positions=point_locations, collision=False, rgba=rgba)


class DropRestingPoseEnv(BasePoseEnv):
    def __init__(self):
        human = Human(controllable_joint_indices=controllable_joints, controllable=True)
        super(BasePoseEnv, self).__init__(human=human)
        self.time_step = 1./240.
        self.steps = 0
        self.max_steps = 200
        self.plane_chair = None
        self.human_pose = None
        self.once = True
