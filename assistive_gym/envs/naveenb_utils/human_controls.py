import time
import math
import numpy as np
from pybullet_utils import pd_controller_stable
import pybullet as p

class HumanStablePD():

    def __init__(self):
        self. stablePD = pd_controller_stable.PDControllerStableMultiDof(p)

        self.kp_org
        self.kd_org

        self.jointIndicesAll = controllable_joints
        self.jointDofCounts = [4,4,4,1,4,4,1,4,1,4,4,1]
        self.total_dofs = 7

        for dof in self.jointDofCounts:
            self.total_dofs += dof

        self.simTimeStep = timeStep

        