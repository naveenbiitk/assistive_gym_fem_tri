
import numpy as np


class stretch_controller():

    def __init__(self):
        #super().__init__(render=render, seed=seed, num_frames=num_frames)
        #angular pid constansts
        self.prev_angle_term = 0
        self.integral_object_angle = 0
        #linear pid constants
        self.prev_linear_term = 0
        self.integral_linear_gap = 0


    def linear_controller(self, object_gap):
        kp_gap = 4.25/1.3
        kd_gap = 0.04
        ki_gap = 0.0000
        
        z = 0
        dt = 0.1
    
        #object_gap = setpoint_x-current_x
        
        derivative_object_gap = (object_gap - prev_linear_term)/dt
        integral_object_gap = integral_linear_gap+object_gap*dt
    
        z = kp_gap*object_gap + kd_gap*derivative_object_gap + ki_gap*integral_object_gap #Remove if object_gap < 0 statement and replace with this line
    
        prev_linear_term = object_gap
        integral_linear_gap = integral_object_gap
    
        return z

    def angular_controller(self, object_angle_gap):
        kp_angle_gap = 4.02/2.5
        kd_angle_gap = 0.00#5
        ki_angle_gap = 0.0000

        z = 0
        dt = 0.1

        #object_angle_gap = setpoint_theta-current_theta
        #object_angle_gap = min(object_angle_gap, 360-object_angle_gap)

        derivative_object_angle_gap = (object_angle_gap - prev_angle_term)/dt
        integral_object_angle_gap = integral_object_angle +object_angle_gap*dt

        z = kp_angle_gap*object_angle_gap + kd_angle_gap*derivative_object_angle_gap + ki_angle_gap*integral_object_angle_gap

        prev_angle_term = object_angle_gap
        integral_object_angle = integral_object_angle_gap
        #print('Ki angle term',ki_angle_gap*integral_object_angle_gap)
        return z



    def position_controller(self, setpoint_position, current_position):

        error_position = np.linalg.norm(np.array(setpoint_position[:2])-np.array(current_position[:2]))
        error_theta = (np.linalg.norm(np.array(setpoint_position[2])-np.array(current_position[2])))%360
        linear_v = 0.0
        angular_v = 0.0
        error_threshold = 0.32
        theta_threshold = 2
        position_threshold = 0.2

        if abs(error_position) > position_threshold or min(abs(error_theta),360-abs(error_theta)) > theta_threshold:
            if abs(error_position) > position_threshold:#position control
                theta_del = math.atan( (np.array(setpoint_position[1])-np.array(current_position[1]))/(np.array(setpoint_position[0])-np.array(current_position[0])) )
                d_del = np.max(np.abs(np.array(setpoint_position[:2])-np.array(current_position[:2])))

                linear_v = linear_controller(d_del)
                angular_v = angular_controller(theta_del)
                #print(' Position change: ', error_position, linear_v)
            elif min(abs(error_theta),360-abs(error_theta)) > theta_threshold:
                print('Angular control')
                angular_v = angular_controller(np.array(setpoint_position[2])-np.array(current_position[2]))
                angular_v = min(angular_v, 0.6)
                angular_v = max(angular_v, -0.6)

        return linear_v,angular_v




# if __name__ == "__main__":
#     c = stretch_sim_real()
#     c.env_render()