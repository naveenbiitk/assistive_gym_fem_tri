# coding: utf-8

import numpy as np
from math import *
import pickle
import json
# import quaternion
from .transformations import *
import torch
from pytorch3d import transforms as T

from pyquaternion import Quaternion


def get_angle(vec1, vec2):
  cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
  return acos(cos_theta)


def get_quaternion(ox, oy, oz, x, y, z):
  # given transformed axis in x-y-z order return a quaternion
  ox /= np.linalg.norm(ox)
  oy /= np.linalg.norm(oy)
  oz /= np.linalg.norm(oz)

  set1 = np.vstack((ox, oy, oz))

  x /= np.linalg.norm(x)
  y /= np.linalg.norm(y)
  z /= np.linalg.norm(z)

  set2 = np.vstack((x, y, z))
  rot_mat = superimposition_matrix(set1, set2, scale=False, usesvd=True)
  rot_qua = quaternion_from_matrix(rot_mat)

  return rot_qua

# ref:https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/deep_mimic/mocap/inverse_kinematics.py
# human_joint_angle_list = [root, chest, neck, r_hip, r_knee, r_ankle, r_should, r_elbow, l_hip, l_knee, l_ankle, l_should, l_elbow]
# 3D coord to deepmimic rotations
def coord_to_rot(frame):
  eps = 0.001
  axis_rotate_rate = 0.3
  hm_to_smpl = np.array([0,2,5,11,1,4,10,3,6,12,15,16,18,20,17,19,21])
  frame = np.array(frame)
  tmp = [[] for i in range(13)]
  # root position (3D),
  root_y = (frame[hm_to_smpl[7]] - frame[hm_to_smpl[0]])
  root_z = (frame[hm_to_smpl[1]] - frame[hm_to_smpl[0]])
  root_x = np.cross(root_y, root_z)

  x = np.array([1.0, 0, 0])
  y = np.array([0, 1.0, 0])
  z = np.array([0, 0, 1.0])

  rot_qua = euler_from_quaternion(get_quaternion(root_x, root_y, root_z, x, y, z))
  tmp[0] = list(rot_qua)

  # chest rotation (4D),
  chest_y = (frame[hm_to_smpl[8]] - frame[hm_to_smpl[7]])
  chest_z = (frame[hm_to_smpl[14]] - frame[hm_to_smpl[8]])
  chest_x = np.cross(chest_y, chest_z)
  rot_qua = euler_from_quaternion(get_quaternion(chest_x, chest_y, chest_z, root_x, root_y, root_z))
  tmp[1] = list(rot_qua)

  # neck rotation (4D),
  neck_y = (frame[hm_to_smpl[10]] - frame[hm_to_smpl[8]])
  neck_z = np.cross(frame[hm_to_smpl[10]] - frame[hm_to_smpl[9]], frame[hm_to_smpl[8]] - frame[hm_to_smpl[9]])
  neck_x = np.cross(neck_y, neck_z)
  rot_qua = euler_from_quaternion(get_quaternion(neck_x, neck_y, neck_z, chest_x, chest_y, chest_z))
  tmp[2] = list(rot_qua)

  # right hip rotation (4D),
  r_hip_y = (frame[hm_to_smpl[1]] - frame[hm_to_smpl[2]])
  r_hip_z = np.cross(frame[hm_to_smpl[1]] - frame[hm_to_smpl[2]], frame[hm_to_smpl[3]] - frame[hm_to_smpl[2]])
  r_hip_x = np.cross(r_hip_y, r_hip_z)
  rot_qua = euler_from_quaternion(get_quaternion(r_hip_x, r_hip_y, r_hip_z, root_x, root_y, root_z))
  tmp[3] = list(rot_qua)

  # right knee rotation (1D),
  vec1 = frame[hm_to_smpl[1]] - frame[hm_to_smpl[2]]
  vec2 = frame[hm_to_smpl[3]] - frame[hm_to_smpl[2]]
  angle1 = get_angle(vec1, vec2)
  tmp[4] = [angle1 - pi]

  # right ankle rotation (4D),
  tmp[5] = euler_from_quaternion([1, 0, 0, 0])

  #  right shoulder rotation (4D),
  r_shou_y = (frame[hm_to_smpl[14]] - frame[hm_to_smpl[15]])
  r_shou_z = np.cross(frame[hm_to_smpl[16]] - frame[hm_to_smpl[15]], frame[hm_to_smpl[14]] - frame[hm_to_smpl[15]])
  r_shou_x = np.cross(r_shou_y, r_shou_z)
  rot_qua = euler_from_quaternion(get_quaternion(r_shou_x, r_shou_y, r_shou_z, chest_x, chest_y, chest_z))
  tmp[6] = list(rot_qua)
  # x is right or left
  # y is axial rotation
  # z is forward backward
  
  # right elbow rotation (1D),
  vec1 = frame[hm_to_smpl[14]] - frame[hm_to_smpl[15]]
  vec2 = frame[hm_to_smpl[16]] - frame[hm_to_smpl[15]]
  angle1 = get_angle(vec1, vec2)
  tmp[7] = [pi - angle1]

  # left hip rotation (4D),
  l_hip_y = (frame[hm_to_smpl[4]] - frame[hm_to_smpl[5]])
  l_hip_z = np.cross(frame[hm_to_smpl[4]] - frame[hm_to_smpl[5]], frame[hm_to_smpl[6]] - frame[hm_to_smpl[5]])
  l_hip_x = np.cross(l_hip_y, l_hip_z)
  rot_qua = euler_from_quaternion(get_quaternion(l_hip_x, l_hip_y, l_hip_z, root_x, root_y, root_z))
  tmp[8] = list(rot_qua)

  # left knee rotation (1D),
  vec1 = frame[hm_to_smpl[4]] - frame[hm_to_smpl[5]]
  vec2 = frame[hm_to_smpl[6]] - frame[hm_to_smpl[5]]
  angle1 = get_angle(vec1, vec2)
  tmp[9] = [angle1 - pi]

  # left ankle rotation (4D),
  tmp[10] = euler_from_quaternion([1, 0, 0, 0])

  # left shoulder rotation (4D),
  l_shou_y = (frame[hm_to_smpl[11]] - frame[hm_to_smpl[12]])
  l_shou_z = np.cross(frame[hm_to_smpl[13]] - frame[hm_to_smpl[12]], frame[hm_to_smpl[11]] - frame[hm_to_smpl[12]])
  l_shou_x = np.cross(l_shou_y, l_shou_z)
  rot_qua = euler_from_quaternion(get_quaternion(l_shou_x, l_shou_y, l_shou_z, chest_x, chest_y, chest_z))
  tmp[11] = list(rot_qua)

  # left elbow rotation (1D)
  vec1 = frame[hm_to_smpl[11]] - frame[hm_to_smpl[12]]
  vec2 = frame[hm_to_smpl[13]] - frame[hm_to_smpl[12]]
  angle1 = get_angle(vec1, vec2)
  tmp[12] = [pi - angle1]

  return tmp


def coord_seq_to_rot_seq(coord_seq, frame_duration):
  ret = []
  for i in range(len(coord_seq)):
    tmp = coord_to_rot(i, coord_seq[i], frame_duration)
    ret.append(list(tmp))
  return ret


# ref: https://github.com/openxrlab/xrmocap/blob/main/xrmocap/transform/rotation/__init__.py
#def body_pose_sja = 
def aa_to_sja(body_pose_reshape):
  R_t = torch.Tensor([
    [[1, 0, 0], [0, 0, 1], [0, -1, 0]],  # 00, 'left_hip',
    [[1, 0, 0], [0, 0, 1], [0, -1, 0]],  # 01, 'right_hip',
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]],  # 02, 'spine1',
    [[1, 0, 0], [0, 0, 1], [0, -1, 0]],  # 03, 'left_knee',
    [[1, 0, 0], [0, 0, 1], [0, -1, 0]],  # 04, 'right_knee',
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]],  # 05, 'spine2',
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # 06, 'left_ankle',
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # 07, 'right_ankle',
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]],  # 08, 'spine3',
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # 09, 'left_foot',
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # 10, 'right_foot',
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]],  # 11, 'neck',
    [[0, 0, -1], [0, 1, 0], [1, 0, 0]],  # 12, 'left_collar',
    [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],  # 13, 'right_collar',
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]],  # 14, 'head',
    [[0, 0, -1], [0, 1, 0], [1, 0, 0]],  # 15, 'left_shoulder',
    [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],  # 16, 'right_shoulder',
    [[0, 0, -1], [0, 1, 0], [1, 0, 0]],  # 17, 'left_elbow',
    [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],  # 18, 'right_elbow',
    [[0, 0, -1], [0, 1, 0], [1, 0, 0]],  # 19, 'left_wrist',
    [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],  # 20, 'right_wrist',
    [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],  # 20, 'right_wrist',
])
  R_t_inv = torch.inverse(R_t)

  R_aa = T.axis_angle_to_matrix(body_pose_reshape)
  R_sja = R_t @ R_aa @ R_t_inv
  sja = T.matrix_to_euler_angles(R_sja, convention='XYZ')
  return sja


STANDARD_JOINT_ANGLE_LIMITS = torch.deg2rad(
    torch.Tensor([
        [[-45, 155], [-88, 17], [-105, 85]],  # 00, 'left_hip',
        [[-45, 155], [-17, 88], [-85, 105]],  # 01, 'right_hip',
        [[-25, 15], [-20, 20], [-30, 30]],  # 02, 'spine1',
        [[0, 150], [-1, 1], [-1, 1]],  # 03, 'left_knee',
        [[0, 150], [-1, 1], [-1, 1]],  # 04, 'right_knee',
        [[-25, 15], [-15, 15], [-25, 25]],  # 05, 'spine2',
        [[-31, 63], [-26, 26], [-74, 15]],  # 06, 'left_ankle',
        [[-31, 63], [-26, 26], [-15, 74]],  # 07, 'right_ankle',
        [[-25, 15], [-15, 15], [-25, 25]],  # 08, 'spine3',
        [[-60, 45], [-1, 1], [-45, 45]],  # 09, 'left_foot',
        [[-60, 45], [-1, 1], [-45, 45]],  # 10, 'right_foot',
        [[-37, 22], [-30, 30], [-45, 45]],  # 11, 'neck',
        [[-30, 30], [-30, 10], [-1, 1]],  # 12, 'left_collar',
        [[-30, 30], [-10, 30], [-1, 1]],  # 13, 'right_collar',
        [[-37, 22], [-30, 30], [-45, 45]],  # 14, 'head',
        [[-135, 90], [-135, 90], [-90, 90]],  # 15, 'left_shoulder',  1 0 2
        [[-135, 90], [-90, 135], [-90, 90]],  # 16, 'right_shoulder',
        [[-1, 1], [-150, 0], [-1, 1]],  # 17, 'left_elbow',
        [[-1, 1], [0, 150], [-1, 1]],  # 18, 'right_elbow',
        [[-90, 90], [-45, 45], [-180, 60]],  # 19, 'left_wrist',
        [[-90, 90], [-45, 45], [-60, 180]],  # 20, 'right_wrist',
    ]))



hs = 0.2
rs=0.1

joint_p, joint_o = [0, 0, 0], [0, 0, 0, 1]
chest_p = [0, 0, 1.148*hs]
base = chest_p
#houlders_p = 

neck_p = [0, 0, 0.132*hs]
head_p = [0, 0, 0.12*hs]
right_upperarm_p = [-0.092*rs - 0.067, 0, 0]
left_upperarm_p = [0.092*rs + 0.067, 0, 0]
forearm_p = [0, 0, -0.264*hs]
hand_p = [0, 0, -(0.027*rs + 0.234*hs)]
waist_p = [0, 0, -0.15*hs]
hips_p = [0, 0, -0.15/2*hs]
right_thigh_p = [-0.0775*rs - 0.0145, 0, -0.15/2*hs]
left_thigh_p = [0.0775*rs + 0.0145, 0, -0.15/2*hs]
shin_p = [0, 0, -0.391*hs]
foot_p = [0, 0, -0.367*hs - 0.045/2]