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


def ik_random_restarts(self, right, target_pos, target_orient, max_iterations=1000, max_ik_random_restarts=40, success_threshold=0.03, step_sim=False, check_env_collisions=False, randomize_limits=True, collision_objects=[]):
        if target_orient is not None and len(target_orient) < 4:
            target_orient = self.get_quaternion(target_orient)
        orient_orig = target_orient
        best_ik_angles = None
        best_ik_distance = 0
        for r in range(max_ik_random_restarts):
            target_joint_angles = self.ik(self.right_end_effector if right else self.left_end_effector, target_pos, target_orient, ik_indices=self.right_arm_ik_indices if right else self.left_arm_ik_indices, max_iterations=max_iterations, half_range=self.half_range, randomize_limits=(randomize_limits and r >= 10))
            self.set_joint_angles(self.right_arm_joint_indices if right else self.left_arm_joint_indices, target_joint_angles)
            gripper_pos, gripper_orient = self.get_pos_orient(self.right_end_effector if right else self.left_end_effector)
            if np.linalg.norm(target_pos - np.array(gripper_pos)) < success_threshold and (target_orient is None or np.linalg.norm(target_orient - np.array(gripper_orient)) < success_threshold or np.isclose(np.linalg.norm(target_orient - np.array(gripper_orient)), 2, atol=success_threshold)):
                # if step_sim:
                #     # TODO: Replace this with getClosestPoints, see: https://github.gatech.edu/zerickson3/assistive-gym/blob/vr3/assistive_gym/envs/feeding.py#L156
                #     for _ in range(5):
                #         p.stepSimulation(physicsClientId=self.id)
                #     # if len(p.getContactPoints(bodyA=self.body, bodyB=self.body, physicsClientId=self.id)) > 0 and orient_orig is not None:
                #     #     # The robot's arm is in contact with itself. Continually randomize end effector orientation until a solution is found
                #     #     target_orient = self.get_quaternion(self.get_euler(orient_orig) + np.deg2rad(self.np_random.uniform(-45, 45, size=3)))
                # if check_env_collisions:
                #     for _ in range(25):
                #         p.stepSimulation(physicsClientId=self.id)

                # Check if the robot is colliding with objects in the environment. If so, then continue sampling.
                if len(collision_objects) > 0:
                    dists_list = []
                    for obj in collision_objects:
                        dists_list.append(self.get_closest_points(obj, distance=0)[-1])
                    if not all(not d for d in dists_list):
                        continue
                gripper_pos, gripper_orient = self.get_pos_orient(self.right_end_effector if right else self.left_end_effector)
                if np.linalg.norm(target_pos - np.array(gripper_pos)) < success_threshold and (target_orient is None or np.linalg.norm(target_orient - np.array(gripper_orient)) < success_threshold or np.isclose(np.linalg.norm(target_orient - np.array(gripper_orient)), 2, atol=success_threshold)):
                    self.set_joint_angles(self.right_arm_joint_indices if right else self.left_arm_joint_indices, target_joint_angles)
                    return True, np.array(target_joint_angles)
            if best_ik_angles is None or np.linalg.norm(target_pos - np.array(gripper_pos)) < best_ik_distance:
                best_ik_angles = target_joint_angles
                best_ik_distance = np.linalg.norm(target_pos - np.array(gripper_pos))
        self.set_joint_angles(self.right_arm_joint_indices if right else self.left_arm_joint_indices, best_ik_angles)
        return False, np.array(best_ik_angles)

    def position_robot_toc(self, task, arms, start_pos_orient, target_pos_orients, human, base_euler_orient=np.zeros(3), max_ik_iterations=200, max_ik_random_restarts=1, randomize_limits=False, attempts=100, jlwki_restarts=1, step_sim=False, check_env_collisions=False, right_side=True, random_rotation=30, random_position=0.5):
        # Continually randomize the robot base position and orientation
        # Select best base pose according to number of goals reached and manipulability
        if type(arms) == str:
            arms = [arms]
            start_pos_orient = [start_pos_orient]
            target_pos_orients = [target_pos_orients]
        a = 6 # Order of the robot space. 6D (3D position, 3D orientation)
        best_position = None
        best_orientation = None
        best_num_goals_reached = None
        best_manipulability = None
        best_start_joint_poses = [None]*len(arms)
        iteration = 0
        # Save human joint states for later restoring
        human_angles = human.get_joint_angles(human.controllable_joint_indices)
        while iteration < attempts or best_position is None:
            iteration += 1
            # Randomize base position and orientation
            random_pos = np.array([self.np_random.uniform(-random_position if right_side else 0, 0 if right_side else random_position), self.np_random.uniform(-random_position, random_position), 0])
            random_orientation = self.get_quaternion([base_euler_orient[0], base_euler_orient[1], base_euler_orient[2] + np.deg2rad(self.np_random.uniform(-random_rotation, random_rotation))])
            self.set_base_pos_orient(np.array([-0.85, -0.4, 0]) + self.toc_base_pos_offset[task] + random_pos, random_orientation)
            # Reset all robot joints to their defaults
            self.reset_joints()
            # Reset human joints in case they got perturbed by previous iterations
            human.set_joint_angles(human.controllable_joint_indices, human_angles)
            num_goals_reached = 0
            manipulability = 0.0
            start_joint_poses = [None]*len(arms)
            # Check if the robot can reach all target locations from this base pose
            for i, arm in enumerate(arms):
                right = (arm == 'right')
                ee = self.right_end_effector if right else self.left_end_effector
                ik_indices = self.right_arm_ik_indices if right else self.left_arm_ik_indices
                lower_limits = self.right_arm_lower_limits if right else self.left_arm_lower_limits
                upper_limits = self.right_arm_upper_limits if right else self.left_arm_upper_limits
                for j, (target_pos, target_orient) in enumerate(start_pos_orient[i] + target_pos_orients[i]):
                    best_jlwki = None
                    best_joint_positions = None
                    for k in range(jlwki_restarts):
                        # Reset state in case anything was perturbed from the last iteration
                        human.set_joint_angles(human.controllable_joint_indices, human_angles)
                        # Find IK solution
                        success, joint_positions_q_star = self.ik_random_restarts(right, target_pos, target_orient, max_iterations=max_ik_iterations, max_ik_random_restarts=max_ik_random_restarts, success_threshold=0.03, step_sim=step_sim, check_env_collisions=check_env_collisions, randomize_limits=randomize_limits)
                        if not success:
                            continue
                        _, motor_positions, _, _ = self.get_motor_joint_states()
                        joint_velocities = [0.0] * len(motor_positions)
                        joint_accelerations = [0.0] * len(motor_positions)
                        center_of_mass = p.getLinkState(self.body, ee, computeLinkVelocity=True, computeForwardKinematics=True, physicsClientId=self.id)[2]
                        J_linear, J_angular = p.calculateJacobian(self.body, ee, localPosition=center_of_mass, objPositions=motor_positions, objVelocities=joint_velocities, objAccelerations=joint_accelerations, physicsClientId=self.id)
                        J_linear = np.array(J_linear)[:, ik_indices]
                        J_angular = np.array(J_angular)[:, ik_indices]
                        J = np.concatenate([J_linear, J_angular], axis=0)
                        # Joint-limited-weighting
                        joint_limit_weight = self.joint_limited_weighting(joint_positions_q_star, lower_limits, upper_limits)
                        # Joint-limited-weighted kinematic isotropy (JLWKI)
                        det = max(np.linalg.det(np.matmul(np.matmul(J, joint_limit_weight), J.T)), 0)
                        jlwki = np.power(det, 1.0/a) / (np.trace(np.matmul(np.matmul(J, joint_limit_weight), J.T))/a)
                        if best_jlwki is None or jlwki > best_jlwki:
                            best_jlwki = jlwki
                            best_joint_positions = joint_positions_q_star
                    if best_jlwki is not None:
                        num_goals_reached += 1
                        manipulability += best_jlwki
                        if j == 0:
                            start_joint_poses[i] = best_joint_positions
                    if j < len(start_pos_orient[i]) and best_jlwki is None:
                        # Not able to find an IK solution to a start goal. We cannot use this base pose
                        num_goals_reached = -1
                        manipulability = None
                        break
                if num_goals_reached == -1:
                    break

            if num_goals_reached > 0:
                if best_position is None or num_goals_reached > best_num_goals_reached or (num_goals_reached == best_num_goals_reached and manipulability > best_manipulability):
                    best_position = random_pos
                    best_orientation = random_orientation
                    best_num_goals_reached = num_goals_reached
                    best_manipulability = manipulability
                    best_start_joint_poses = start_joint_poses

            human.set_joint_angles(human.controllable_joint_indices, human_angles)

        # Reset state in case anything was perturbed
        human.set_joint_angles(human.controllable_joint_indices, human_angles)

        # Set the robot base position/orientation and joint angles based on the best pose found
        p.resetBasePositionAndOrientation(self.body, np.array([-0.85, -0.4, 0]) + np.array(self.toc_base_pos_offset[task]) + best_position, best_orientation, physicsClientId=self.id)
        for i, arm in enumerate(arms):
            self.set_joint_angles(self.right_arm_joint_indices if arm == 'right' else self.left_arm_joint_indices, best_start_joint_poses[i])
        return best_position, best_orientation, best_start_joint_poses

    def joint_limited_weighting(self, q, lower_limits, upper_limits):
        phi = 0.5
        lam = 0.05
        weights = []
        for qi, l, u in zip(q, lower_limits, upper_limits):
            qr = 0.5*(u - l)
            weights.append(1.0 - np.power(phi, (qr - np.abs(qr - qi + l)) / (lam*qr) + 1))
            if weights[-1] < 0.001:
                weights[-1] = 0.001
        # Joint-limited-weighting
        joint_limit_weight = np.diag(weights)
        return joint_limit_weight
        

# see mobile phone environemnt is working or not

# import mobile phone environemnt

# env_name is ShowPhoneStretchEnv-v1

# jwlki  outputs possible starting base location of stretch-->optimize?--->outputs starting position
# We have fixed starting point want to find goal point?


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












if __name__ == "__main__":

    # viewer(args.env)
    #pose_drop_viewer()
    # optimization_viewer('DropRestingPose-v1')
    #optimization_viewer('OptimizationTest-v1')
    test_optimization_viewer('OptimizationTest-v1')