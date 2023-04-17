import pybullet as p
from math import pi


def build_chair(seat_height=0.4,
                seat_width=0.3,
                seat_depth=0.3,
                left_arm_height=0.65,
                right_arm_height=0.65,
                arm_width=0.075,
                arm_length=0.25,
                arm_separation=0.75,
                back_angle=1.4,
                feature_thickness=0.01,
                chair_color=[0.6, 0.6, 0.6, 1.]):

    # Build Seat
    seat_size = [seat_width, seat_depth, feature_thickness]
    base_collision = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                            halfExtents=seat_size)

    base_visual = p.createVisualShape(shapeType=p.GEOM_BOX,
                                      halfExtents=seat_size,
                                      rgbaColor=chair_color)

    # Build Arms
    arm_size = [arm_length, arm_width, feature_thickness]
    left_arm_collision = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                halfExtents=arm_size)

    left_arm_visual = p.createVisualShape(shapeType=p.GEOM_BOX,
                                          halfExtents=arm_size,
                                          rgbaColor=chair_color)

    right_arm_collision = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                 halfExtents=arm_size)

    right_arm_visual = p.createVisualShape(shapeType=p.GEOM_BOX,
                                           halfExtents=arm_size,
                                           rgbaColor=chair_color)

    # Build Back
    back_size = [seat_width, seat_width, feature_thickness]
    back_collision = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                            halfExtents=back_size)

    back_visual = p.createVisualShape(shapeType=p.GEOM_BOX,
                                      halfExtents=back_size,
                                      rgbaColor=chair_color)

    back_orientation = p.getQuaternionFromEuler((0, back_angle, 0.))

    # Build pybullet objects
    seat = p.createMultiBody(baseCollisionShapeIndex=base_collision,
                             baseVisualShapeIndex=base_visual,
                             basePosition=(0., 0., seat_height))

    left_arm = p.createMultiBody(baseCollisionShapeIndex=left_arm_collision,
                                 baseVisualShapeIndex=left_arm_visual,
                                 basePosition=(0., arm_separation/2., left_arm_height))

    right_arm = p.createMultiBody(baseCollisionShapeIndex=right_arm_collision,
                                  baseVisualShapeIndex=right_arm_visual,
                                  basePosition=(0., -arm_separation/2., right_arm_height))

    back = p.createMultiBody(baseCollisionShapeIndex=back_collision,
                             baseVisualShapeIndex=back_visual,
                             basePosition=(-0.2, 0., seat_height + 0.3),
                             baseOrientation=back_orientation)

    output = [seat, left_arm, right_arm, back]
    return output