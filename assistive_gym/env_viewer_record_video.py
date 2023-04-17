import gym, sys, argparse
import numpy as np
#from .learn_human import make_env
from .learn_human import make_env
# import assistive_gym
import imageio_ffmpeg

if sys.version_info < (3, 0):
    print('Please use Python 3')
    exit()

import cv2
import os


def save_video_clip(env, filename='test_1', num_episodes=1, fps=10, coop=True):

    if not os.path.exists('videos'):
        os.makedirs('videos')

    env_width = 1280
    env_height = 1620

    vid = imageio_ffmpeg.write_frames(filename, (env_width, env_height), fps=30)
    vid.send(None) # seed the video writer with a blank frame
    action_list = []
    observation_list = []
    for i in range(num_episodes):
        done = False
        observation = env.reset()
        count = 0
        print('---Saving video clip', i+1, 'of', num_episodes, 'to', filename, '...')
        while count<50:
            action = sample_action(env, coop)
            observation, reward, done, info = env.step(action)
            action_list.append(action)
            observation_list.append(observation)
            img_d,depth = env.get_camera_image_depth()
            img = img_d[:,:,0:3]
            #print('--img got', img.shape)
            #img = cv2.cvtColor( img, cv2.COLOR_BGR2RGB )
            #print(img.shape)
            vid.send(np.ascontiguousarray(img))
            #cv2.imshow('img', img)
            count += 1
        vid.close()
        print('---Video saved to', filename)
        # save the action and observation in .npz file
        np.savez('videos/test_2.npz', action_list=action_list, observation_list=observation_list)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #save the image
    cv2.imwrite('videos/test_2.png', img)
    print('---Video saved to', filename)
    vid.close()
    print('---Video saved to', filename)
    # save the action and observation in .npz file
    np.savez('videos/test_2.npz', action_list=action_list, observation_list=observation_list)


def sample_action(env, coop=True):
    if coop:
        return {'robot': env.action_space_robot.sample(), 'human': env.action_space_human.sample()}
    return env.action_space.sample()

def sample_action_human(env, coop):
    #if coop:
    #    return {'robot': env.action_space_robot.sample(), 'human': env.action_space_human.sample()}
    return env.action_space_human.sample()



def viewer(env_name):
    coop = 'Human' in env_name
    env = make_env(env_name, coop=True) if coop else gym.make(env_name)

    #save_video_clip(env, filename='videos/'+env_name+'_test_2'+'.mp4', num_episodes=1, fps=10, coop=coop)
    while True:
        done = False
        env.render()
        observation = env.reset()
        #action = sample_action_human(env, coop)
        
        # if coop:
        #     print('Robot observation size:', np.shape(observation['robot']), 'Human observation size:', np.shape(observation['human']), 'Robot action size:', np.shape(action['robot']), 'Human action size:', np.shape(action['human']))
        # else:
        #     print('Observation size:', np.shape(observation), 'Action size:', np.shape(action))

        while not done:
            action = sample_action_human(env, coop)
            #action = sample_action(env, coop)
            observation, reward, done, info = env.step(action)
            if coop:
                #print(done)
                done = 0
                #done = done['__all__']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assistive Gym Environment Viewer')
    parser.add_argument('--env', default='RobotMotionStretch-v1-v1',
                        help='Environment to test (default: ScratchItchJaco-v1)')
    args = parser.parse_args()

    viewer(args.env)



# def viewer_backup(env_name):
#     coop = 'Human' in env_name
#     env = make_env(env_name, coop=True) if coop else gym.make(env_name)
#     env.render()

#     save_video_clip(env, filename='videos/'+env_name+'.mp4', num_episodes=1, fps=30)

#     while True:
#         done = False
#         env.render()
#         observation = env.reset()
#         action = sample_action(env, coop)
#         # if coop:
#         #     print('Robot observation size:', np.shape(observation['robot']), 'Human observation size:', np.shape(observation['human']), 'Robot action size:', np.shape(action['robot']), 'Human action size:', np.shape(action['human']))
#         # else:
#         #     print('Observation size:', np.shape(observation), 'Action size:', np.shape(action))

#         while not done:
#             observation, reward, done, info = env.step(sample_action(env, coop))
#             if coop:
#                 #print(done)
#                 done = 0
#                 #done = done['__all__']