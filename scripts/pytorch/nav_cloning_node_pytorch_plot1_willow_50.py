#!/usr/bin/env python3
from __future__ import print_function

from numpy import dtype
import roslib
roslib.load_manifest('nav_cloning')
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_cloning_pytorch import *
from skimage.transform import resize
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseArray
from std_srvs.srv import Trigger
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_srvs.srv import SetBool, SetBoolResponse
import csv
import os
import copy
import sys
import tf
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Bool, String
import time


class nav_cloning_node:
    def __init__(self):
        rospy.init_node('nav_cloning_node', anonymous=True)
        self.mode = rospy.get_param("/nav_cloning_node/mode", "change_dataset_balance")
        self.action_num = 1
        self.dl = deep_learning(n_action = self.action_num)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        self.image_left_sub = rospy.Subscriber("/camera_left/rgb/image_raw", Image, self.callback_left_camera)
        self.image_right_sub = rospy.Subscriber("/camera_right/rgb/image_raw", Image, self.callback_right_camera)
        self.vel_sub = rospy.Subscriber("/nav_vel", Twist, self.callback_vel)
        self.nav_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.vel_pub = rospy.Publisher('/pie_vel', Float32, queue_size=1)
        self.srv = rospy.Service('/training', SetBool, self.callback_dl_training)
        self.mode_save_srv = rospy.Service('/model_save', Trigger, self.callback_model_save)
        self.pose_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.callback_pose)
        self.path_sub = rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.callback_path)
        # self.path_sub = rospy.Subscriber("/move_base/GlobalPlanner/plan", Path, self.callback_path)
        self.min_distance = 0.0
        self.action = 0.0
        self.episode = 0
        self.vel = Twist()
        self.path_pose = PoseArray()
        self.cv_image = np.zeros((480,640,3), np.uint8)
        self.cv_left_image = np.zeros((480,640,3), np.uint8)
        self.cv_right_image = np.zeros((480,640,3), np.uint8)
        self.learning = True
        self.select_dl = False
        self.path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/result_'+str(self.mode)+'/'
        self.save_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/model_'+str(self.mode)+'/'
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_the = 0.0
        self.is_started = False
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.previous_reset_time = 0
        self.start_time_s = rospy.get_time()
        os.makedirs(self.path + self.start_time)
        self.tracker_sub = rospy.Subscriber("/tracker", Odometry, self.callback_tracker)
        self.exit_sub = rospy.Subscriber("/learning_exit", Bool, self.callback_exit)
        self.exit_flg = False
        self.kill_flg = False
        self.kill_count = 0
        self.dir_pub = rospy.Publisher('/dir', String, queue_size=1)
        self.mode_pub = rospy.Publisher("/mode", Bool, queue_size=1)
        self.episode_csv_flg = 0


    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_left_camera(self, data):
        try:
            self.cv_left_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_right_camera(self, data):
        try:
            self.cv_right_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_tracker(self, data):
        self.pos_x = data.pose.pose.position.x
        self.pos_y = data.pose.pose.position.y
        rot = data.pose.pose.orientation
        angle = tf.transformations.euler_from_quaternion((rot.x, rot.y, rot.z, rot.w))
        self.pos_the = angle[2]

    def callback_path(self, data):
        self.path_pose = data

    def callback_pose(self, data):
        distance_list = []
        pos = data.pose.pose.position
        for pose in self.path_pose.poses:
            path = pose.pose.position
            distance = np.sqrt(abs((pos.x - path.x)**2 + (pos.y - path.y)**2))
            distance_list.append(distance)

        if distance_list:
            self.min_distance = min(distance_list)


    def callback_vel(self, data):
        self.vel = data
        self.action = self.vel.angular.z

    def callback_dl_training(self, data):
        resp = SetBoolResponse()
        self.learning = data.data
        resp.message = "Training: " + str(self.learning)
        resp.success = True
        return resp

    def callback_model_save(self, data):
        model_res = SetBoolResponse()
        self.dl.save(self.save_path)
        model_res.message ="model_save"
        model_res.success = True
        return model_res

    def callback_exit(self, data):
        self.exit_flg = data.data


    def loop(self):
        if self.cv_image.size != 640 * 480 * 3:
            return
        if self.cv_left_image.size != 640 * 480 * 3:
            return
        if self.cv_right_image.size != 640 * 480 * 3:
            return
        if self.vel.linear.x != 0:
            self.is_started = True
        if self.is_started == False:
            return

        self.dir_pub.publish(self.path + self.start_time)

        img = resize(self.cv_image, (48, 64), mode='constant')
        img_left = resize(self.cv_left_image, (48, 64), mode='constant')
        img_right = resize(self.cv_right_image, (48, 64), mode='constant')


        if self.exit_flg:
            self.exit_flg = False
            self.learning = False
            self.dl.save(self.save_path)
            self.kill_flg = True
            

        if self.kill_flg:
            self.kill_count += 1
            if self.kill_count == 2500:
                os.system('killall roslaunch')
                sys.exit()

        if self.episode == 8000:
            self.learning = False
            self.dl.save(self.save_path)
            # self.dl.load("/home/yuzuki//model_gpu.pt")

        mode = self.learning
        self.mode_pub.publish(mode)


        if self.episode == 10000:
            os.system('killall roslaunch')
            sys.exit()


        if self.learning:
            target_action = self.action
            distance = self.min_distance

            if self.mode == "follow_line":
                action, loss = self.dl.act_and_trains(img , target_action)
                if abs(target_action) < 0.1:
                    action_left,  loss_left  = self.dl.act_and_trains(img_left , target_action - 0.2)
                    action_right, loss_right = self.dl.act_and_trains(img_right , target_action + 0.2)
                angle_error = abs(action - target_action)



            elif self.mode == "use_dl_output":
                action, loss = self.dl.act_and_trains(img , target_action)
                if abs(target_action) < 0.1:
                    action_left,  loss_left  = self.dl.act_and_trains(img_left , target_action - 0.2)
                    action_right, loss_right = self.dl.act_and_trains(img_right , target_action + 0.2)
                angle_error = abs(action - target_action)
                if distance > 0.1:
                    self.select_dl = False
                elif distance < 0.05:
                    self.select_dl = True
                if self.select_dl and self.episode >= 0:
                    target_action = action




            elif self.mode == "change_dataset_balance":
                if distance < 0.05:
                    action, loss = self.dl.act_and_trains(img , target_action)
                    if abs(target_action) < 0.1:
                        action_left,  loss_left  = self.dl.act_and_trains(img_left , target_action - 0.2)
                        action_right, loss_right = self.dl.act_and_trains(img_right , target_action + 0.2)
                elif 0.05 <= distance < 0.1:
                    self.dl.make_dataset(img , target_action)
                    action, loss = self.dl.act_and_trains(img , target_action)
                    if abs(target_action) < 0.1:
                        self.dl.make_dataset(img_left , target_action - 0.2)
                        action_left,  loss_left  = self.dl.act_and_trains(img_left , target_action - 0.2)
                        self.dl.make_dataset(img_right , target_action + 0.2)
                        action_right, loss_right = self.dl.act_and_trains(img_right , target_action + 0.2)
                    lines = [str(self.episode), "training", str(distance), str(self.pos_x), str(self.pos_y), str(self.pos_the)  ]
                    with open(self.path + self.start_time + '/' + 'training_all.csv', 'a') as f:
                        writer = csv.writer(f, lineterminator='\n')
                        writer.writerow(lines)
                   
                else:
                    self.dl.make_dataset(img , target_action)
                    self.dl.make_dataset(img , target_action)
                    action, loss = self.dl.act_and_trains(img , target_action)
                    if abs(target_action) < 0.1:
                        self.dl.make_dataset(img_left , target_action - 0.2)
                        self.dl.make_dataset(img_left , target_action - 0.2)
                        action_left,  loss_left  = self.dl.act_and_trains(img_left , target_action - 0.2)
                        self.dl.make_dataset(img_right , target_action + 0.2)
                        self.dl.make_dataset(img_right , target_action + 0.2)
                        action_right, loss_right = self.dl.act_and_trains(img_right , target_action + 0.2)
                    lines = [str(self.episode), "training", str(distance), str(self.pos_x), str(self.pos_y), str(self.pos_the)  ]
                    with open(self.path + self.start_time + '/' + 'training_all.csv', 'a') as f:
                        writer = csv.writer(f, lineterminator='\n')
                        writer.writerow(lines)
                    with open(self.path + self.start_time + '/' + 'training_all.csv', 'a') as f:
                        writer = csv.writer(f, lineterminator='\n')
                        writer.writerow(lines)


                angle_error = abs(action - target_action)
                if distance > 0.1:
                    self.select_dl = False
                elif distance < 0.05:
                    self.select_dl = True
                if self.select_dl and self.episode >= 0:
                    target_action = action


            # end mode
           
            self.episode += 1
            print(str(self.episode) + ", training, loss: " + str(loss) + ", angle_error: " + str(angle_error) + ", distance: " + str(distance))
            lines = [str(self.episode), "training", str(distance), str(self.pos_x), str(self.pos_y), str(self.pos_the)  ]
            with open(self.path + self.start_time + '/' + 'training_all.csv', 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(lines)

            self.vel.linear.x = 0.2
            if angle_error < 0.02:
                with open(self.path + self.start_time + '/' + 'training.csv', 'a') as f:
                    writer = csv.writer(f, lineterminator='\n')
                    writer.writerow(lines)
            self.vel.linear.x = 0.2
            self.vel.angular.z = target_action
            self.vel_pub.publish(self.vel.angular.z)
            self.nav_pub.publish(self.vel)
            

        if self.learning == False:
            if self.episode_csv_flg == 0:
                episode = [str(self.episode)]
                with open(self.path + self.start_time + '/' + 'learning_exit_episode.csv', 'a') as f:
                    writer = csv.writer(f, lineterminator='\n')
                    writer.writerow(episode)
                self.episode_csv_flg = 1
            target_action = self.dl.act(img)
            distance = self.min_distance
            print(str(self.episode) + ", test, angular:" + str(target_action) + ", distance: " + str(distance))
            self.episode += 1
            angle_error = abs(self.action - target_action)
            lines = [str(self.episode), "test", str(distance), str(self.pos_x), str(self.pos_y), str(self.pos_the)  ]
            with open(self.path + self.start_time + '/' + 'training_all.csv', 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(lines)
            if angle_error < 0.02:
                with open(self.path + self.start_time + '/' + 'training.csv', 'a') as f:
                    writer = csv.writer(f, lineterminator='\n')
                    writer.writerow(lines)
            self.vel.linear.x = 0.2
            self.vel.angular.z = target_action
            self.vel_pub.publish(self.vel.angular.z)
            self.nav_pub.publish(self.vel)

       
        temp = copy.deepcopy(img)
        cv2.imshow("Resized Image", temp)
        temp = copy.deepcopy(img_left)
        cv2.imshow("Resized Left Image", temp)
        temp = copy.deepcopy(img_right)
        cv2.imshow("Resized Right Image", temp)
        cv2.waitKey(1)

if __name__ == '__main__':
    rg = nav_cloning_node()
    DURATION = 0.2
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()