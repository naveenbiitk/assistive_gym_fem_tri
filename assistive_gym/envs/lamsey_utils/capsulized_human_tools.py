import matplotlib.pyplot as plt
from statistics import mean
import pybullet as p
import numpy as np


def get_articulated_joint_indices(human):
    lower_limits = np.array([value for key, value in human.lower_limits.items()])
    upper_limits = np.array([value for key, value in human.upper_limits.items()])
    joint_ranges = upper_limits - lower_limits
    articulated_joint_indices = [i for i, range in enumerate(joint_ranges) if range > 0.]
    return articulated_joint_indices


def get_chin_pos_orient(env, bool_plot=False):
    head_pos, head_orient = env.human.get_pos_orient(link=env.human.head)

    # get head mesh data
    head_link = 23  # from human_creation.py
    head_mesh = p.getMeshData(env.human.body, head_link)  # undocumented function
    head_mesh = head_mesh[1]
    x = [pt[0] for pt in head_mesh]
    y = [pt[1] for pt in head_mesh]
    z = [pt[2] for pt in head_mesh]

    # chin location thresholds - determined by observation
    chin_x_min = -0.02
    chin_x_max = 0.02
    chin_y_min = -0.10
    chin_y_max = -0.075
    chin_z_min = -0.01
    chin_z_max = 0.01

    # get points near chin
    indices = range(len(x))
    chin_indices = [i for i in indices if
                    chin_x_max > x[i] > chin_x_min and chin_y_max > y[i] > chin_y_min and chin_z_max > z[
                        i] > chin_z_min]
    chin_x = [x[i] for i in chin_indices]
    chin_y = [y[i] for i in chin_indices]
    chin_z = [z[i] for i in chin_indices]

    if bool_plot:
        head_x = [x[i] for i in indices if i not in chin_indices]
        head_y = [y[i] for i in indices if i not in chin_indices]
        head_z = [z[i] for i in indices if i not in chin_indices]

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        # ax.scatter(x, y, z)
        ax.scatter(head_x, head_y, head_z)
        ax.scatter(chin_x, chin_y, chin_z)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    # transform local chin coordinates to global frame
    chin_position = (mean(chin_x), mean(chin_y), mean(chin_z))
    chin_position_global, chin_orientation_global = p.multiplyTransforms(positionA=head_pos, orientationA=head_orient,
                                                   positionB=chin_position, orientationB=[0., 0., 0., 1])

    return chin_position_global, chin_orientation_global