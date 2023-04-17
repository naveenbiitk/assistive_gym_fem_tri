from gym.envs.registration import register

tasks = ['ScratchItch', 'BedBathing', 'Feeding', 'Drinking', 'Dressing', 'ArmManipulation']
robots = ['PR2', 'Jaco', 'Baxter', 'Sawyer', 'Stretch', 'Panda']

for task in tasks:
    for robot in robots:
        register(
            id='%s%s-v1' % (task, robot),
            entry_point='assistive_gym.envs:%s%sEnv' % (task, robot),
            max_episode_steps=200,
        )

for task in ['ScratchItch', 'Feeding']:
    for robot in robots:
        register(
            id='%s%sMesh-v1' % (task, robot),
            entry_point='assistive_gym.envs:%s%sMeshEnv' % (task, robot),
            max_episode_steps=200,
        )

register(
    id='HumanTesting-v1',
    entry_point='assistive_gym.envs:HumanTestingEnv',
    max_episode_steps=200,
)

register(
    id='HumanStanding-v1',
    entry_point='assistive_gym.envs:HumanStandingEnv',
    max_episode_steps=200,
)

register(
    id='HumanResting-v1',
    entry_point='assistive_gym.envs:HumanRestingEnv',
    max_episode_steps=200,
)

register(
    id='HumanLying-v1',
    entry_point='assistive_gym.envs:HumanLyingEnv',
    max_episode_steps=200,
)

register(
    id='SMPLXTesting-v1',
    entry_point='assistive_gym.envs:SMPLXTestingEnv',
    max_episode_steps=200,
)

register(
    id='StretchTesting-v1',
    entry_point='assistive_gym.envs:StretchTestingEnv',
    max_episode_steps=200,
)

register(
    id='JointReachingStretch-v1', 
    entry_point='assistive_gym.envs:JointReachingStretchEnv',
    max_episode_steps=200,
)

register(
    id='JointReachingPR2-v1', 
    entry_point='assistive_gym.envs:JointReachingPR2Env',
    max_episode_steps=200,
)

register(
    id='ObjectHandoverStretch-v1', 
    entry_point='assistive_gym.envs:ObjectHandoverStretchEnv',
    max_episode_steps=200,
)

register(
    id='ObjectHandoverStretchHuman-v1', 
    entry_point='assistive_gym.envs:ObjectHandoverStretchHumanEnv',
    max_episode_steps=200,
)

register(
    id='ObjectHandoverPR2-v1', 
    entry_point='assistive_gym.envs:ObjectHandoverPR2Env',
    max_episode_steps=200,
)

register(
    id='RobotMotionStretch-v1', 
    entry_point='assistive_gym.envs:RobotMotionStretchEnv',
    max_episode_steps=200,
)

register(
    id='JointMotionStretch-v1', 
    entry_point='assistive_gym.envs:JointMotionStretchEnv',
    max_episode_steps=200,
)

register(
    id='JointMotionPR2-v1', 
    entry_point='assistive_gym.envs:JointMotionPR2Env',
    max_episode_steps=200,
)

register(
    id='ShowPhoneStretch-v1', 
    entry_point='assistive_gym.envs:ShowPhoneStretchEnv',
    max_episode_steps=200,
)

register(
    id='ShowPhonePR2-v1', 
    entry_point='assistive_gym.envs:ShowPhonePR2Env',
    max_episode_steps=200,
)


register(
    id='TestTask-v1',
    entry_point='assistive_gym.envs:TestEnv',
    max_episode_steps=1,
)

register(
    id='ControlTest-v1',
    entry_point='assistive_gym.envs:ControlPR2Env',
    max_episode_steps=200
)

register(
    id='OptimizationTest-v1',
    entry_point='assistive_gym.envs:OptimizationEnv',
    max_episode_steps=200
)

register(
    id='StableRestingPose-v1',
    entry_point='assistive_gym.envs:StableRestingPoseEnv',
    max_episode_steps=100,
)

register(
    id='DropRestingPose-v1',
    entry_point='assistive_gym.envs:DropRestingPoseEnv',
    max_episode_steps=999999,
)

register(
    id='TestTask-v1',
    entry_point='assistive_gym.envs:TestEnv',
    max_episode_steps=1,
)

register(
    id='ControlTest-v1',
    entry_point='assistive_gym.envs:ControlPR2Env',
    max_episode_steps=200
)

register(
    id='OptimizationTest-v1',
    entry_point='assistive_gym.envs:OptimizationEnv',
    max_episode_steps=200
)

register(
    id='HumanSitcane-v1',
    entry_point='assistive_gym.envs:HumanSitcaneEnv',
    max_episode_steps=200,
)