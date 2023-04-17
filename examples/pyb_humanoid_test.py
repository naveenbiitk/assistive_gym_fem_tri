import pybullet as p
import time
import numpy as np

import pybullet_data

width = 1920
height = 1080
p.connect(p.GUI, options='--background_color_red=0.8 --background_color_green=0.9 --background_color_blue=1.0 --width=%d --height=%d' % (width, height))
p.setAdditionalSearchPath(pybullet_data.getDataPath())
obUids = p.loadMJCF("mjcf/humanoid.xml")#,  flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)

orient_ = [0,0,3.14/2]
orient_q = p.getQuaternionFromEuler(orient_)

chairs = [p.loadURDF("assets/bed/hospital_bed.urdf", -0.500000, 0.000000, 0.000000, orient_q[0], orient_q[1], orient_q[2], orient_q[3]) ]


humanoid = obUids[1]

gravId = p.addUserDebugParameter("gravity", -10, 10, -10)
jointIds = []
paramIds = []

numJoints = p.getNumJoints(humanoid)
for j in range (numJoints):
  p.setJointMotorControl2(humanoid,j, p.VELOCITY_CONTROL,force=0)

p.changeDynamics(humanoid,-1,linearDamping=0.9)

#p.setPhysicsEngineParameter(numSolverIterations=10)
#p.changeDynamics(humanoid, -1, linearDamping=0, angularDamping=0)

        #for j in self.all_joint_indices:
        #    p.setJointMotorControl2(self.body, jointIndex=j, controlMode=p.VELOCITY_CONTROL, force=0, physicsClientId=self.id)



jdict = {}
for j in range(p.getNumJoints(humanoid)):
  p.setJointMotorControl2(humanoid, jointIndex=j, controlMode=p.VELOCITY_CONTROL, force=0)
  p.changeDynamics(humanoid, j, linearing=1.1, angularDamping=0.1)
  info = p.getJointInfo(humanoid, j)
  #print(info)
  jointName = info[1]

  jointType = info[2]
  if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
    #print(jointName)
    jname = info[1].decode("ascii")
    jdict[jname] = j
    jointIds.append(j)
    paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -15, 15, 0))


special_joints = ["abdomen_z", "abdomen_y", "abdomen_x","right_hip_x", "right_hip_z", "right_hip_y",]

motor_names = ["abdomen_z", "abdomen_y", "abdomen_x"]
motor_power = [100, 100, 100]
motor_names += ["right_hip_x", "right_hip_z", "right_hip_y", "right_knee"]
motor_power += [100, 100, 300, 200]
motor_names += ["right_ankle_y", "right_ankle_x"]
motor_power += [0, 0]
motor_names += ["left_hip_x", "left_hip_z", "left_hip_y", "left_knee"]
motor_power += [100, 100, 300, 200]
motor_names += ["left_ankle_y", "left_ankle_x"]
motor_power += [0, 0]
motor_names += ["right_shoulder1", "right_shoulder2", "right_elbow"]
motor_power += [75, 75, 75]
motor_names += ["left_shoulder1", "left_shoulder2", "left_elbow"]
motor_power += [75, 75, 75]
motors = [jdict[n] for n in motor_names]

p.setRealTimeSimulation(1)
i_=0
while (i_<500):
  p.setGravity(0, 0, p.readUserDebugParameter(gravId))
  for i in range(len(paramIds)):
    c = paramIds[i]
    targetPos = p.readUserDebugParameter(c)
    p.setJointMotorControl2(humanoid, jointIds[i], p.POSITION_CONTROL, targetPos, force=5 * 240.)
  time.sleep(0.01)
  i_ = i_+1


print('----torque control-------')

force_base_arr = []
for j in range(p.getNumJoints(humanoid)):
  jointState = p.getJointStateMultiDof(humanoid, j)
  if j in jointIds:
    print("jointStateMultiDof[", j, "].jointForces=", jointState[3][0])
    force_base_arr.append(jointState[3][0])

for j in range (numJoints):
  info = p.getJointInfo(humanoid, j)
  #print(info)
  jointName = info[1]
  
  if jointName not in special_joints:
    p.setJointMotorControl2(humanoid,j, p.VELOCITY_CONTROL,force=0)

forces = [0.] * len(motors)

force_base = [-3.7534455895423893, -3.1266925215721133, 8.02051385641098, 22.57042226791382, -7.6829188704490665, 45.29503891468048, -11.755613422393798, 0.0, 0.0, -13.790477561950684, -5.526544582843781, -67.60092673301698, 16.629321384429932, 0.0, 0.0, -3.1006616324186327, 2.3434491947293283, -0.26267900168895725, -2.3656905591487885, 0.2426534526050091, -2.3395349755883217]



while (1):
  p.setGravity(0, 0, p.readUserDebugParameter(gravId))
  read_arr = []
  for i in range(len(paramIds)):
    c = paramIds[i]
    targetPos = p.readUserDebugParameter(c)
    read_arr.append(targetPos)
    #p.setJointMotorControl2(humanoid, jointIds[i], p.TORQUE_CONTROL, targetPos*100, force=5 * 240.)

  for m in range(len(motors)):
    limit = 15
    ac = np.clip(read_arr[m], -limit, limit)
    #print(ac)  
    forces[m] = force_base_arr[m] + motor_power[m] * ac * 0.082
  
  #print('Forces',forces)  
  p.setJointMotorControlArray(humanoid, jointIds, controlMode=p.POSITION_CONTROL, forces=forces)

  time.sleep(0.01)
