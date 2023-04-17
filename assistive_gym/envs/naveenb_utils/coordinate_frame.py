import pybullet as p
import numpy as np
from math import pi

def human_pose_smpl_format(human_i):

    human_pt_position = np.zeros((20,3))
    human_pt_orientation = np.zeros((20,4))
    #human_pt_position
    human_pt_position[0,:], human_pt_orientation[0,:] = human_i.get_pos_orient(human_i.head)
    human_pt_position[1,:], human_pt_orientation[1,:] = human_i.get_pos_orient(human_i.neck)
    human_pt_position[2,:], human_pt_orientation[2,:] = human_i.get_pos_orient(human_i.stomach)
    human_pt_position[3,:], human_pt_orientation[3,:] = human_i.get_pos_orient(human_i.waist)
    human_pt_position[4,:], human_pt_orientation[4,:] = human_i.get_pos_orient(human_i.left_pecs)
    human_pt_position[5,:], human_pt_orientation[5,:] = human_i.get_pos_orient(human_i.right_pecs)
    #arms
    human_pt_position[6,:], human_pt_orientation[6,:] = human_i.get_pos_orient(human_i.right_shoulder)
    human_pt_position[7,:], human_pt_orientation[7,:] = human_i.get_pos_orient(human_i.left_shoulder)
    human_pt_position[8,:], human_pt_orientation[8,:] = human_i.get_pos_orient(human_i.j_right_forearm)
    human_pt_position[9,:], human_pt_orientation[9,:] = human_i.get_pos_orient(human_i.j_left_forearm)
    human_pt_position[10,:], human_pt_orientation[10,:] = human_i.get_pos_orient(human_i.j_right_wrist_x)
    human_pt_position[11,:], human_pt_orientation[11,:] = human_i.get_pos_orient(human_i.j_left_wrist_x)
    human_pt_position[12,:], human_pt_orientation[12,:] = human_i.get_pos_orient(human_i.right_elbow)
    human_pt_position[13,:], human_pt_orientation[13,:] = human_i.get_pos_orient(human_i.left_elbow)
    #legs
    human_pt_position[14,:], human_pt_orientation[14,:] = human_i.get_pos_orient(human_i.right_hip)
    human_pt_position[15,:], human_pt_orientation[15,:] = human_i.get_pos_orient(human_i.left_hip)
    human_pt_position[16,:], human_pt_orientation[16,:] = human_i.get_pos_orient(human_i.right_knee)
    human_pt_position[17,:], human_pt_orientation[17,:] = human_i.get_pos_orient(human_i.left_knee)
    human_pt_position[18,:], human_pt_orientation[18,:] = human_i.get_pos_orient(human_i.right_ankle)
    human_pt_position[19,:], human_pt_orientation[19,:] = human_i.get_pos_orient(human_i.left_ankle)

    return human_pt_position, human_pt_orientation



# take these 20 poses draw line update every single time
# 


# return human_pt_position, human_pt_orientation
# n = len(human_pt_position)

# for i in range(n):
#   generate_line_hand(human_pt_position[i], human_pt_orientation[i])



def generate_line_hand(pos, orient, lineLen=0.5):
    
    mat = p.getMatrixFromQuaternion(orient)
    dir0 = [mat[0], mat[3], mat[6]]
    dir1 = [mat[1], mat[4], mat[7]]
    dir2 = [mat[2], mat[5], mat[8]]
    
    # works only for hand 0.25 linelen
    dir2_neg = [-mat[2], -mat[5], -mat[8]]
    to1 = [pos[0] + lineLen * (dir2_neg[0]+dir0[0])/2, pos[1] + lineLen * (dir2_neg[1]+dir0[1])/2, pos[2] + lineLen * (dir2_neg[2]+dir0[2])/2 ]
    to2 = [pos[0] + lineLen * (dir2_neg[0]-dir0[0])/2, pos[1] + lineLen * (dir2_neg[1]-dir0[1])/2, pos[2] + lineLen * (dir2_neg[2]+dir0[2])/2 ]
    
    # works only for head  1.5 linlen
    # dir2_neg = [-mat[1], -mat[4], -mat[7]]
    # to1 = [pos[0] + lineLen * (dir2_neg[0]+dir0[0])/2, pos[1] + lineLen * (dir2_neg[1]+dir0[1])/2, pos[2] + lineLen * (dir2_neg[2]+dir0[2])/2 ]
    # to2 = [pos[0] + lineLen * (dir2_neg[0]-dir0[0])/2, pos[1] + lineLen * (dir2_neg[1]-dir0[1])/2, pos[2] + lineLen * (dir2_neg[2]+dir0[2])/2 ]
    
    toX = [pos[0] + lineLen * dir0[0], pos[1] + lineLen * dir0[1], pos[2] + lineLen * dir0[2]]
    toY = [pos[0] + lineLen * dir1[0], pos[1] + lineLen * dir1[1], pos[2] + lineLen * dir1[2]]
    toZ = [pos[0] + lineLen * dir2[0], pos[1] + lineLen * dir2[1], pos[2] + lineLen * dir2[2]]
    
    p.addUserDebugLine(pos, toX, [1, 0, 0], 5)
    p.addUserDebugLine(pos, toY, [0, 1, 0], 5)
    p.addUserDebugLine(pos, toZ, [0, 0, 1], 5)

    # p.addUserDebugLine(pos, to1, [0, 1, 1], 5, 3)
    # p.addUserDebugLine(pos, to2, [0, 1, 1], 5, 3)
    # p.addUserDebugLine(to2, to1, [0, 1, 1], 5, 3)
