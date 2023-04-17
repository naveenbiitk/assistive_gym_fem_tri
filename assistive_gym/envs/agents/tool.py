import os
import pybullet as p
import numpy as np
from .agent import Agent

class Tool(Agent):
    def __init__(self):
        super(Tool, self).__init__()

    def init(self, robot, task, directory, id, np_random, right=True, mesh_scale=[1]*3, maximal=False, alpha=1.0, mass=1):
        self.robot = robot
        self.task = task
        self.right = right
        self.id = id
        
        print('Task is', task)
        print(robot.body, robot.right_tool_joint, self.body)
        transform_pos, transform_orient = self.get_transform()
        print('transform_pos: ', transform_pos, transform_orient)
        transform_pos = [0,0,0]
        transform_orient=[0,0,0,1]
        # Instantiate the tool mesh
        if task == 'scratch_itch':
            tool = p.loadURDF(os.path.join(directory, 'scratcher', 'tool_scratch.urdf'), basePosition=transform_pos, baseOrientation=transform_orient, physicsClientId=id)
        elif task == 'bed_bathing':
            tool = p.loadURDF(os.path.join(directory, 'bed_bathing', 'wiper.urdf'), basePosition=transform_pos, baseOrientation=transform_orient, physicsClientId=id)
        elif task == 'joint_reaching':
            tool = p.loadURDF(os.path.join(directory, 'scratcher', 'tool_scratch.urdf'), basePosition=transform_pos, baseOrientation=transform_orient, physicsClientId=id)
        elif task == 'joint_motion':
            mass = 0.1
            mesh_scale = [0.05] * 3
            visual_filename = os.path.join(directory, 'dinnerware', 'plastic_coffee_cup.obj')
            collision_filename = os.path.join(directory, 'dinnerware', 'plastic_coffee_cup_vhacd.obj')
            tool_visual = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=visual_filename, meshScale=mesh_scale, rgbaColor=[1, 1, 1, alpha], physicsClientId=id)
            tool_collision = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=collision_filename, meshScale=mesh_scale, physicsClientId=id)
            transform_pos = np.array([-0.2, -0.81, 0.78]) 
            tool = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=tool_collision, baseVisualShapeIndex=tool_visual, basePosition=transform_pos, baseOrientation=transform_orient, useMaximalCoordinates=maximal, physicsClientId=id)

        elif task == 'show_phone':
            tool = p.loadURDF(os.path.join(directory, 'scratcher', 'tool_scratch.urdf'), basePosition=transform_pos, baseOrientation=transform_orient, physicsClientId=id)
 
        elif task == 'robot_motion':
            mass = 0.1
            mesh_scale = [0.05] * 3
            transform_orient = p.getQuaternionFromEuler([np.pi/2, np.pi/2, np.pi/2])
            visual_filename = os.path.join(directory, 'dinnerware', 'plastic_coffee_cup.obj')
            collision_filename = os.path.join(directory, 'dinnerware', 'plastic_coffee_cup_vhacd.obj')
            tool_visual = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=visual_filename, meshScale=mesh_scale, rgbaColor=[1, 1, 1, alpha], physicsClientId=id)
            tool_collision = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=collision_filename, meshScale=mesh_scale, physicsClientId=id)
            transform_pos = np.array([-0.2, -0.53, 0.78]) 
            tool = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=tool_collision, baseVisualShapeIndex=tool_visual, basePosition=transform_pos, baseOrientation=transform_orient, useMaximalCoordinates=maximal, physicsClientId=id)

        elif task == 'object_handover':
            #tool = p.loadURDF(os.path.join(directory, 'scratcher', 'WalkingCane.urdf'), basePosition=transform_pos, baseOrientation=transform_orient, physicsClientId=id)
            #tool = p.loadSDF(os.path.join(directory, 'scratcher', 'WalkingCane.sdf'))
            mass = 0.1
            mesh_scale = [1.05] * 3
            visual_filename = os.path.join(directory, 'scratcher', 'WalkingCane.obj')
            collision_filename = os.path.join(directory, 'scratcher', 'WalkingCane.obj')
            tool_visual = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=visual_filename, meshScale=mesh_scale, physicsClientId=id)
            tool_collision = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=collision_filename, meshScale=mesh_scale, physicsClientId=id)
            transform_pos = np.array([-0.2, -0.81, 0.08]) 
            tool = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=tool_collision, baseVisualShapeIndex=tool_visual, basePosition=transform_pos, baseOrientation=transform_orient, useMaximalCoordinates=maximal, physicsClientId=id)

        elif task in ['drinking', 'feeding', 'arm_manipulation']:
            if task == 'drinking':
                visual_filename = os.path.join(directory, 'dinnerware', 'plastic_coffee_cup.obj')
                collision_filename = os.path.join(directory, 'dinnerware', 'plastic_coffee_cup_vhacd.obj')
            elif task == 'feeding':
                visual_filename = os.path.join(directory, 'dinnerware', 'spoon.obj')
                collision_filename = os.path.join(directory, 'dinnerware', 'spoon_vhacd.obj')
            elif task == 'arm_manipulation':
                visual_filename = os.path.join(directory, 'arm_manipulation', 'arm_manipulation_scooper.obj')
                collision_filename = os.path.join(directory, 'arm_manipulation', 'arm_manipulation_scooper_vhacd.obj')
            tool_visual = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=visual_filename, meshScale=mesh_scale, rgbaColor=[1, 1, 1, alpha], physicsClientId=id)
            tool_collision = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=collision_filename, meshScale=mesh_scale, physicsClientId=id)
            tool = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=tool_collision, baseVisualShapeIndex=tool_visual, basePosition=transform_pos, baseOrientation=transform_orient, useMaximalCoordinates=maximal, physicsClientId=id)
        else:
            tool = None

        if task == 'human_sitcane':
            #transform_pos = [0,0,0]
            self.pos_offset = [0,0,-0.7]
            tool = p.loadURDF(os.path.join(directory, 'scratcher', 'WalkingCane.urdf'), basePosition=transform_pos, baseOrientation=transform_orient, physicsClientId=id)            
            #self.orient_offset = self.get_quaternion([0,0,0])
            #self.pos_offset = [0,0,0]
            self.body = tool
            constraint = p.createConstraint(robot.body, robot.right_tool_joint, self.body, -1, p.JOINT_FIXED, [0, 0, 0], parentFramePosition=self.pos_offset, childFramePosition=[0, 0, 0], parentFrameOrientation=self.orient_offset, childFrameOrientation=[0, 0, 0, 1], physicsClientId=id)
            p.changeConstraint(constraint, maxForce=1000, physicsClientId=id)


        super(Tool, self).init(tool, id, np_random, indices=-1)


        #comment if the tool needs to be gripped by the robot
        if robot is not None and task is not 'human_sitcane' and task is not 'robot_motion':
            #Disable collisions between the tool and robot
            for j in (robot.right_gripper_collision_indices if right else robot.left_gripper_collision_indices):
                for tj in self.all_joint_indices + [self.base]:
                    p.setCollisionFilterPair(robot.body, self.body, j, tj, False, physicsClientId=id)
            # Create constraint that keeps the tool in the gripper
            # Commented the two lines for putting the tool down
            constraint = p.createConstraint(robot.body, robot.right_tool_joint if right else robot.left_tool_joint, self.body, -1, p.JOINT_FIXED, [0, 0, 0.0], parentFramePosition=self.pos_offset, childFramePosition=[0, 0, 0.0], parentFrameOrientation=self.orient_offset, childFrameOrientation=[0, 0, 0, 1], physicsClientId=id)
            p.changeConstraint(constraint, maxForce=500, physicsClientId=id)

    def get_transform(self):
        
        if self.task == 'human_sitcane':
            gripper_pos, gripper_orient = self.robot.get_pos_orient(self.robot.right_tool_joint)
            # transform_pos = gripper_pos
            # gripper_orient = [0,0,0,1]
            #self.pos_offset = [0,-0.15,-0.5] #this is for cane constraint
            self.pos_offset = [0,-0.0,-0.0]
            self.orient_offset = self.get_quaternion([0,0,np.pi])
            transform_pos, transform_orient = p.multiplyTransforms(positionA=gripper_pos, orientationA=gripper_orient, positionB=self.pos_offset, orientationB=self.orient_offset, physicsClientId=self.id)
            # print('transform_pos: ', transform_pos, transform_orient)
            transform_pos = [0,0,0]#gripper_pos
            transform_orient = [0,0,0,1]
            return transform_pos, transform_orient #this is for loading the object

        if self.robot is not None:
            self.pos_offset = self.robot.tool_pos_offset[self.task]
            self.orient_offset = self.get_quaternion(self.robot.tool_orient_offset[self.task])
            self.orient_offset = self.get_quaternion([np.pi/2,0,0])
            gripper_pos, gripper_orient = self.robot.get_pos_orient(self.robot.right_tool_joint if self.right else self.robot.left_tool_joint, center_of_mass=True)
            transform_pos, transform_orient = p.multiplyTransforms(positionA=gripper_pos, orientationA=gripper_orient, positionB=self.pos_offset, orientationB=self.orient_offset, physicsClientId=self.id)
        else:
            transform_pos = [0, 0, 0]
            transform_orient = [0, 0, 0, 1]
        return transform_pos, transform_orient

    def reset_pos_orient(self):
        transform_pos, transform_orient = self.get_transform()
        self.set_base_pos_orient(transform_pos, transform_orient)

