import pybullet as p
import pybullet_data
from assistive_gym.learn import make_env
import numpy as np




def calculate_obj_ik(env, target_pose_angles, target_position):

  mid_angle = np.array([-1.28484584e-14,  1.74532925e-01,  1.24455499e-14, -8.98027383e-02, -1.10828408e+00,  
              6.82478992e-14, -1.11701072e+00, -1.43938821e-14,  7.85398163e-02,  1.74532925e-01])
  
  max_moment = 363.4084744
  max_ang = 16.7190253038

  weights_opt_ik = np.array([ 0.5,  0.5,  0.1 ])

  forces = np.array([env.human.get_force_torque_sensor(joints_j) for joints_j in env.human.right_arm_joints])

  linear_forces = np.sqrt( forces[:,0]*forces[:,0] + forces[:,1]*forces[:,1] + forces[:,2]*forces[:,2] )
  linear_moment = np.sqrt( forces[:,3]*forces[:,3] + forces[:,4]*forces[:,4] + forces[:,5]*forces[:,5] )
  
  f_moment = np.sum(linear_moment**2)
  f_moment = f_moment/(max_moment**2)

  f_angle = np.sum( (target_pose_angles-mid_angle)**2 )
  f_angle = f_angle/(max_ang**2)

  current_position = env.human.get_pos_orient(env.human.right_wrist)[0]
  f_position = (target_position-current_position)**2

  f_final = (f_moment*weights_opt_ik[0] + f_angle*weights_opt_ik[1] + f_position*weights_opt_ik[2])/(np.sum(weights_opt_ik))
  
  return f_final



def position_human_arm(env, target_position, attempts):
  
  human_angles = env.human.get_joint_angles(env.human.right_arm_joints)
  iteration = 0

  best_sol = None
  best_joint_positions = None

  while iteration < attempts:
    iteration += 1
    target_joint_angles_h = env.human.ik(env.human.right_wrist, target_position, None, ik_indices=env.human.right_arm_joints, max_iterations=2000)

    env.human.set_joint_angles(env.human.right_arm_joints, target_joint_angles_h)

    f_sol = calculate_obj_ik(env, target_joint_angles_h, target_position)

    if best_sol is None or f_sol < best_sol:
      best_sol = f_sol
      best_joint_positions = target_joint_angles_h

    return best_joint_positions




def run_simulation(env, target_position, count_target=150):

    #env.target_pos = np.array(target_position)
    env.create_sphere(radius=0.02, mass=0.0, pos= target_position, visual=True, collision=False, rgba=[1, 1, 0, 1]) 
    target_joint_angles = env.human.ik(env.human.right_wrist, target_position, None, ik_indices=env.human.right_arm_joints, max_iterations=2000)
    target_joint_angles_h = position_human_arm(env, target_position, attempts=100)
    count_ = 0
    
    while count_<count_target:
        
        env.render()
        #print('--Joint angles', target_pose_angles)
        #print(env.human.right_arm_joints)
        env.human.set_joint_angles(env.human.right_arm_joints, target_joint_angles_h)
        #if coop:
        #    action ={'robot': env.action_space_robot.sample(), 'human': env.action_space_human.sample()}
        human_action = np.zeros([6])
        observation, reward, done, info = env.step(human_action)

        count_=count_+1
 


if __name__ == "__main__":
    env_name = "HumanTesting-v1"
    env = make_env(env_name, coop=True)
    # env = gym.make()
    env.render()
    observation = env.reset()
    base_pos_set = [5,-5,0.1]
    quat_orient = env.get_quaternion([0.0, 0.0, 3.14])
    #env.robot.set_base_pos_orient( base_pos_set, quat_orient)

    #p.removeBody(1 , physicsClientId=env.id)
    #test_set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # right arm shoulder and elbow
    #optimizer(env)
    points = [ -0.322725220284914, -0.15785464645505627, 0.80928032059021836]
    #points = [0.18, -0.4, 0.1]  #best one
    #points = [-0.08, -0.25, -0.25]
    run_simulation(env, points)




p.disconnect()