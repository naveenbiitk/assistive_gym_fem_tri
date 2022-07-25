import pybullet as p
from math import pi


def origin_coordinate_system():
    origin = [0., 0., 0.]
    rotation_matrix = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    coordinate_system(origin, rotation_matrix, alpha=0.5)


def coordinate_system(origin, rotation_matrix, axis_length=0.5, axis_radius=0.025, alpha=1.):
    print("coordinate_frame::coordinate_frame: rotation_matrix not implemented!")
    # X Axis
    x_axis = p.createVisualShape(p.GEOM_CYLINDER,
                                 radius=axis_radius,
                                 length=axis_length,
                                 rgbaColor=[1., 0., 0., alpha])

    x_axis_position = [sum(x) for x in zip(origin, [axis_length/2., 0., 0.])]
    x_axis_orientation = p.getQuaternionFromEuler([0., pi/2., 0.])

    # Y Axis
    y_axis = p.createVisualShape(p.GEOM_CYLINDER,
                                 radius=axis_radius,
                                 length=axis_length,
                                 rgbaColor=[0., 1., 0., alpha])

    y_axis_position = [sum(x) for x in zip(origin, [0., axis_length/2., 0.])]
    y_axis_orientation = p.getQuaternionFromEuler([pi / 2., 0., 0.])

    # Z Axis
    z_axis = p.createVisualShape(p.GEOM_CYLINDER,
                                 radius=axis_radius,
                                 length=axis_length,
                                 rgbaColor=[0., 0., 1., alpha])

    z_axis_position = [sum(x) for x in zip(origin, [0., 0., axis_length/2.])]
    z_axis_orientation = p.getQuaternionFromEuler([0., 0., 0.])

    # Create bodies
    x = p.createMultiBody(baseVisualShapeIndex=x_axis,
                          basePosition=x_axis_position,
                          baseOrientation=x_axis_orientation)

    y = p.createMultiBody(baseVisualShapeIndex=y_axis,
                          basePosition=y_axis_position,
                          baseOrientation=y_axis_orientation)

    z = p.createMultiBody(baseVisualShapeIndex=z_axis,
                          basePosition=z_axis_position,
                          baseOrientation=z_axis_orientation)

    return x, y, z