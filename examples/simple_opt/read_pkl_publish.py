# Read a pkl file and publish the point on the ROS publishers

import rospy
from geometry_msgs.msg import Point

import pickle
import numpy as np

from geometry_msgs.msg import Pose, PoseStamped
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray

class PosePublisher:
    def __init__(self, file_pkl_name):

        frame_topic_name = 'frame_pose'
        task_topic_name = 'task_pose'
        self.frame_pub = rospy.Publisher(frame_topic_name, PoseStamped, queue_size=10)
        self.task_pub = rospy.Publisher(task_topic_name, PoseStamped, queue_size=10)
        self.joints_pub = rospy.Publisher('bp_node/joint_points', PoseArray, queue_size=5)
        self.f_name = file_pkl_name
        self.pose = None
        self.task_pose = Pose()
        self.frame_pose = Pose()
        self.pose_array = np.array([])
        self.read_the_pkl_file()

    def set_pose(self, pose):
        self.frame_pose = pose
        self.task_pose = pose

    def read_the_pkl_file(self):
        with open(self.f_name, 'rb') as f:
            file_data = pickle.load(f)
        
        self.score = file_data['score']
        self.point_f = file_data['point']
        self.task_point = file_data['task_point']
        self.task_orient = file_data['task_orient']
        self.frame_point = file_data['frame_point']
        self.frame_orient = file_data['frame_orient']

        self.task_pose.position.x = self.task_point[0]
        self.task_pose.position.y = self.task_point[1]
        self.task_pose.position.z = self.task_point[2]

        self.task_pose.orientation.x = self.task_orient[0]
        self.task_pose.orientation.y = self.task_orient[1]
        self.task_pose.orientation.z = self.task_orient[2]
        self.task_pose.orientation.w = self.task_orient[2]

        self.frame_pose.position.x = self.frame_point[0]
        self.frame_pose.position.y = self.frame_point[1]
        self.frame_pose.position.z = self.frame_point[2]

        self.frame_pose.orientation.x = self.frame_orient[0]
        self.frame_pose.orientation.y = self.frame_orient[1]
        self.frame_pose.orientation.z = self.frame_orient[2]
        self.frame_pose.orientation.w = self.frame_orient[2]

        self.file_2_name = 'examples/optimal_frame_lying/data/smpl_bp_ros_smpl.pkl'
        with open(self.file_2_name, 'rb') as f:
            file_data_2 = pickle.load(f)
        
        self.pose_array = file_data_2['human_joints_3D_est']

    def publish_pose(self):

        task_pose_msg = PoseStamped()
        task_pose_msg.header.stamp = rospy.Time.now()
        task_pose_msg.header.frame_id = "world" # Change this to your desired frame id

        task_pose_msg.pose = self.task_pose
        self.task_pub.publish(task_pose_msg)

        frame_pose_msg = PoseStamped()
        frame_pose_msg.header.stamp = rospy.Time.now()
        frame_pose_msg.header.frame_id = "world" # Change this to your desired frame id

        frame_pose_msg.pose = self.frame_pose
        self.frame_pub.publish(frame_pose_msg)

        pose_array_data = []
        pub_msg = PoseArray()
        pub_msg.header = Header() 
        pub_msg.header.stamp =  rospy.Time.now()
        for i in range(np.shape(self.pose_array)[0]):
            pose_array_data.append( Pose(Point(self.pose_array[i,0],self.pose_array[i,1],self.pose_array[i,2]), Quaternion(0,0,0,1)) )
        
        pub_msg.poses = pose_array_data
        self.joints_pub.publish(pub_msg)
        


    def main_loop(self):
          
        rate = rospy.Rate(2)
	    #rospy.spin()
        while not rospy.is_shutdown():
            self.publish_pose()
            rate.sleep()


if __name__ == '__main__':
    
    try:
        rospy.init_node('body_pose_publisher', anonymous=True)      
        publisher_cl = PosePublisher(file_pkl_name= 'result_realtime_handover.pkl').main_loop()
        #publisher_cl = PosePublisher(file_pkl_name= 'result_realtime_showscreen.pkl').main_loop()
    except rospy.ROSInterruptException:
	    rospy.loginfo('interrupt received, so shutting down')

