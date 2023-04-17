'''
import gym, assistive_gym
import pybullet as p
import numpy as np
import time

import pybullet_data
from assistive_gym.learn_human import make_env

env_name = "HumanStanding-v1"
env = make_env(env_name, coop=True)
#env = gym.make()
env.render()
observation = env.reset()

while True:
  env.render()
  human_action = np.zeros(env.action_human_len)
  #robot_action = np.zeros(env.action_robot_len)
  #final_action =  {'robot': robot_action*20, 'human': human_action*80}
  #angle_p = env.human.get_joint_angles(indices=[25])
  #print('angle', angle_p/3.14*180)
  new_obs, rew, done, info = env.step(human_action)
  for i in range(len(env.paramIds)):
    c = env.paramIds[i]
    targetPos = p.readUserDebugParameter(c)
    p.setJointMotorControl2(env.human.body, env.jointIds[i], p.POSITION_CONTROL, targetPos, force=5 * 240.)
  time.sleep(0.01)


'''
import pybullet as p
import time

import pybullet_data

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.loadSDF("stadium.sdf")
orient_ = [0,0,3.14/2]
orient_q = p.getQuaternionFromEuler(orient_)

chairs = [p.loadURDF("assets/chair/chair.urdf", -0.500000, 0.000000, 0.000000, orient_q[0], orient_q[1], orient_q[2], orient_q[3]) ]

obUids = p.loadMJCF("mjcf/humanoid_symmetric_no_ground.xml")
humanoid = obUids[0]

gravId = p.addUserDebugParameter("gravity", -10, 10, -10)
jointIds = []
paramIds = []

p.setPhysicsEngineParameter(numSolverIterations=10)


numJoints = p.getNumJoints(humanoid)
for j in range (numJoints):
  p.setJointMotorControl2(humanoid,j, p.VELOCITY_CONTROL,force=0)

p.changeDynamics(humanoid,-1,linearDamping=0.9)


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


for j in range(p.getNumJoints(humanoid)):
  #p.changeDynamics(humanoid, j, linearDamping=0, angularDamping=0)
  info = p.getJointInfo(humanoid, j)
  #print(info)
  jointName = info[1]
  jointType = info[2]
  if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
    jointIds.append(j)
    paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -4, 4, 0))

p.setRealTimeSimulation(1)
while (1):
  p.setGravity(0, 0, p.readUserDebugParameter(gravId))
  for i in range(len(paramIds)):
    c = paramIds[i]
    targetPos = p.readUserDebugParameter(c)
    p.setJointMotorControl2(humanoid, jointIds[i], p.POSITION_CONTROL, targetPos, force=5 * 240.)


  time.sleep(0.01)