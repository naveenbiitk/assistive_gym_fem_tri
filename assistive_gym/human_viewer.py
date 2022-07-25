import gym, sys, argparse
import numpy as np
from assistive_gym.learn import make_env

import matplotlib.pyplot as plt

import envs.lamsey_utils.capsulized_human_tools as cht
from scipy.optimize import minimize, basinhopping
import cma
from time import sleep


def action_ik(env):
    right_hand_pos, right_hand_orient = env.human.get_pos_orient(link=env.human.right_wrist)
    head_pos, head_orient = env.human.get_pos_orient(link=env.human.head)
    ik_solution = env.human.ik(target_joint=env.human.right_wrist, target_pos=head_pos, target_orient=right_hand_orient,
                      ik_indices=env.human.right_arm_joints, max_iterations=200)

    joint_angles = env.human.get_joint_angles(env.human.right_arm_joints)

    error = ik_solution - joint_angles
    kp = [0.1] * len(error)
    control_force = kp * error

    action = [0.] * len(env.human.controllable_joint_indices)
    # action[0] = 10.
    for i in range(len(env.human.right_arm_joints)):
        action[env.human.right_arm_joints[i]] = control_force[i]

    return action


def action_drop(env):
    kp = [0.5] * len(env.human.controllable_joint_indices)
    # kp[0] = 10.
    # kp[1] = 10.
    # kp[2] = 10.
    if env.human_pose is not None:
        pose_error = env.human_pose - env.human.get_joint_angles()
        return kp * pose_error[env.human.controllable_joint_indices]
    else:
        return [0.] * len(env.human.controllable_joint_indices)


def action_perturb(env):
    pass


def pose_drop_viewer():
    pose_generation_env_name = 'StableRestingPose-v1'
    pose_drop_env_name = 'DropRestingPose-v1'

    # human_pose = generation_viewer(pose_generation_env_name)
    obs_history = drop_viewer(pose_drop_env_name)

    # analyze observation
    poses = [obs["human_pose"] for obs in obs_history]
    d_poses = np.abs(np.diff(poses, axis=0))
    d_poses_sum = np.array([np.sum(d_pose) for d_pose in d_poses])

    threshold = 0.01
    below_threshold = d_poses_sum[d_poses_sum <= threshold]
    below_threshold_i = [idx for idx, val in enumerate(d_poses_sum) if val <= threshold]
    above_threshold = d_poses_sum[d_poses_sum > threshold]
    above_threshold_i = [idx for idx, val in enumerate(d_poses_sum) if val > threshold]

    # plot
    plt.plot(below_threshold_i, below_threshold)
    plt.plot(above_threshold_i, above_threshold)
    plt.grid()
    plt.show()


def test_optimization_viewer(env_name):
    env = gym.make(env_name)
    env.render()
    env.reset()

    def single_joint_optimization(joint_angle, human, joint_index, bool_sleep=False):
        env.human.set_joint_angles([joint_index], [joint_angle])
        lower_limit = human.lower_limits[joint_index]
        upper_limit = human.upper_limits[joint_index]
        joint_range = upper_limit - lower_limit

        if joint_range > 0:
            joint_center = lower_limit + (joint_range / 2.)
            distance_from_center = joint_angle - joint_center
            distance_normalized = distance_from_center / joint_range
            penalty = abs(distance_normalized)
        else:
            penalty = 0.

        if bool_sleep:
            sleep(0.1)

        return penalty

    def multi_joint_optimization(joint_angles, human, joint_indices, bool_sleep=False):
        env.human.set_joint_angles(joint_indices, joint_angles)
        lower_limits = np.array([human.lower_limits[ind] for ind in joint_indices])
        upper_limits = np.array([human.upper_limits[ind] for ind in joint_indices])
        joint_ranges = upper_limits - lower_limits
        joint_centers = lower_limits + (joint_ranges / 2.)
        current_joint_angles = np.array(human.get_joint_angles(joint_indices))
        distance_from_center = current_joint_angles - joint_centers
        distance_from_center_normalized = distance_from_center / joint_ranges
        penalty = sum(abs(distance_from_center_normalized))
        if bool_sleep:
            sleep(0.1)
        return penalty

    # cht.get_articulated_joint_indices(env.human)
    # test_set = [10, 11, 12, 13, 14, 15, 16]  # right arm shoulder and elbow
    test_set = cht.get_articulated_joint_indices(env.human)
    # test_set = env.human.right_arm_joints + env.human.left_arm_joints
    # test_set = env.human.right_arm_joints
    # test_set = env.human.all_joint_indices

    x0 = env.human.get_joint_angles(test_set)
    # res = minimize(multi_joint_optimization, x0, args=(env.human, test_set), method='Nelder-Mead')
    # res = basinhopping(multi_joint_optimization, x0, minimizer_kwargs={"args": (env.human, test_set), "method": 'Nelder-Mead'})
    # print(res)

    es = cma.CMAEvolutionStrategy(x0, 0.1)
    es.optimize(multi_joint_optimization, args=(env.human, test_set))
    # res = es.result()
    es.result_pretty()
    input()

    return
    methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP']
    success_rate = dict()

    print('')
    print('TESTING SCIPY OPTIMIZATION METHODS FOR SINGLE JOINT')
    print('')
    for method in methods:
        n_successes = 0
        env.reset()
        for i in env.human.all_joint_indices:
            x0 = env.human.get_joint_angles([i])
            res = minimize(single_joint_optimization, x0, args=(env.human, i), method=method)

            if res['success']:
                n_successes += 1

        success_rate.update({method: n_successes})

    # print results
    n_joints = len(env.human.all_joint_indices)
    for key, value in success_rate.items():
        print(key + ": " + str(value) + "/" + str(n_joints))

    print('')
    input()


def optimization_viewer(env_name):
    env = gym.make(env_name)

    action_history = []
    obs_history = []

    pose_optimized = False
    env.render()
    observation = env.reset()

    right_arm_joints = env.human.right_arm_joints

    right_hand_pos, right_hand_orient = env.human.get_pos_orient(link=env.human.right_wrist)
    chin_position_global, chin_orientation_global = cht.get_chin_pos_orient(env)

    ik_solution = env.human.ik(target_joint=env.human.right_wrist, target_pos=chin_position_global, target_orient=right_hand_orient,
                               ik_indices=env.human.right_arm_joints, max_iterations=200)

    # env.create_spheres(radius=0.01, batch_positions=[chin_position_global], collision=False, rgba=[1., 0., 0., 1.])

    env.human.set_joint_angles(right_arm_joints, ik_solution)
    input()

    simulation_finished = False
    while not simulation_finished:
        observation, reward, simulation_finished, info = env.step([0.] * len(env.human.controllable_joint_indices))

    # while not pose_optimized:
    #     joint_angles = env.human.get_joint_angles(right_arm_joints)
    #     env.human.set_joint_angles(right_arm_joints, joint_angles + 0.00001)


def generation_viewer(env_name):
    env = gym.make(env_name)

    action_history = []
    obs_history = []

    simulation_finished = False
    env.render()
    observation = env.reset()
    action = action_ik(env)

    # Output observation and action sizes
    print('Observation size:', np.shape(observation), 'Action size:', np.shape(action))

    while not simulation_finished:
        action = action_ik(env)
        observation, reward, simulation_finished, info = env.step(action)

        action_history.append(action)
        obs_history.append(observation)
        # print(observation)

    return obs_history
    # waist_index = human.torso_joints[3]
    # waist_history = [obs[waist_index] for obs in joint_history]
    # head_pos_history = [j[0] for j in joint_history]

    # n_episodes += 1
    # plt.plot(head_pos_history)
    # # plt.ylim(-math.pi/2, math.pi/2)
    # plt.ylim(-1., 1.5)
    # plt.legend(["Head X", "Head Y", "Head Z"])
    # plt.grid(color="lightgray")
    # plt.ylabel("Position (m)")
    # plt.xlabel("Simulation Step")
    # plt.show()

    # plt.plot(action_history)
    # plt.show()


def drop_viewer(env_name):
    env = gym.make(env_name)

    action_history = []
    obs_history = []

    simulation_finished = False
    env.render()
    observation = env.reset()
    action = action_drop(env)

    # Output observation and action sizes
    print('Observation size:', np.shape(observation), 'Action size:', np.shape(action))

    pose_reached = False

    while not simulation_finished:
        # check if human has reached target stable pose
        if observation["human"]:
            pose_reached = True

        # select action based on phase in process
        if pose_reached:
            action = action_drop(env)
            simulation_finished = True
        else:
            action = action_ik(env)

        action = action_drop(env) if pose_reached else action_ik(env)
        observation, reward, simulation_finished, info = env.step(action)

        action_history.append(action)
        obs_history.append(observation)
        # print(observation)

    return obs_history



if __name__ == "__main__":

    # viewer(args.env)
    #pose_drop_viewer()
    # optimization_viewer('DropRestingPose-v1')
    #optimization_viewer('OptimizationTest-v1')
    test_optimization_viewer('OptimizationTest-v1')