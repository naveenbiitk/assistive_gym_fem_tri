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
    id='SMPLXTesting-v1',
    entry_point='assistive_gym.envs:SMPLXTestingEnv',
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
    id='ObjectHandoverPR2-v1', 
    entry_point='assistive_gym.envs:ObjectHandoverPR2Env',
    max_episode_steps=200,
)