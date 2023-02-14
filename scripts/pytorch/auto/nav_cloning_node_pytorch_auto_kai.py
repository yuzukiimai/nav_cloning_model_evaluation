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
from std_msgs.msg import Int8
from std_srvs.srv import Trigger
from nav_msgs.msg import Path
from std_msgs.msg import Int8MultiArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_srvs.srv import Empty
from std_srvs.srv import SetBool, SetBoolResponse
import csv
import os
import time
import copy
import sys
import tf
from nav_msgs.msg import Odometry
import random

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
        self.action_pub = rospy.Publisher("action", Int8, queue_size=1)
        self.nav_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.srv = rospy.Service('/training', SetBool, self.callback_dl_training)
        self.mode_save_srv = rospy.Service('/model_save', Trigger, self.callback_model_save)
        self.pose_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.callback_pose)
        self.path_sub = rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.callback_path)
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
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/result_'+str(self.mode)+'/'
        self.save_path = roslib.packages.get_pkg_dir('nav_cloning') + '/data/model_'+str(self.mode)+'/'
        self.previous_reset_time = 0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_the = 0.0
        self.is_started = False
        self.start_time_s = rospy.get_time()
        os.makedirs(self.path + self.start_time)
        self.numbers_1 = [1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
        self.numbers_2 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        self.i = 0
        self.mean_training = []

        # with open(self.path + self.start_time + '/' +  'training.csv', 'w') as f:
        #     writer = csv.writer(f, lineterminator='\n')
        #     writer.writerow(['step', 'mode', 'loss', 'angle_error(rad)', 'distance(m)','x(m)','y(m)', 'the(rad)', 'direction'])
        self.tracker_sub = rospy.Subscriber("/tracker", Odometry, self.callback_tracker)

   
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


        if self.episode <= 4000:
            img_hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
            img_hsv_left = cv2.cvtColor(self.cv_left_image, cv2.COLOR_BGR2HSV)
            img_hsv_right = cv2.cvtColor(self.cv_right_image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(img_hsv)
            h_left, s_left, v_left = cv2.split(img_hsv_left)
            h_right, s_right, v_right = cv2.split(img_hsv_right)
            # mean = np.mean(v)
            # mean_left = np.mean(v_left)
            # mean_right = np.mean(v_right)
            # self.mean_training.append(mean)
            # mean_csv=[str(self.episode), mean]
            # with open(self.path + self.start_time + '/' + 'mean.csv', 'a') as f:
            #     writer = csv.writer(f, lineterminator='\n')
            #     writer.writerow(mean_csv)
            gamma = 0.5
            look_up_table = np.zeros((256, 1) ,dtype=np.uint8)
            for i in range(256):
                look_up_table[i][0] = (i/255)**(1.0/gamma)*255
            v_lut = cv2.LUT(v, look_up_table)
            v_lut_left = cv2.LUT(v_left, look_up_table)
            v_lut_right = cv2.LUT(v_right, look_up_table)
            mean = np.mean(v_lut)
            self.mean_training.append(mean)
            mean_csv=[str(self.episode), mean]
            with open(self.path + self.start_time + '/' + 'mean.csv', 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(mean_csv)
            merge = cv2.merge([h, s, v_lut])
            merge_left = cv2.merge([h_left, s_left, v_lut_left])
            merge_right = cv2.merge([h_right, s_right, v_lut_right])
                # print(np.mean(v_lut))   
            bgr = cv2.cvtColor(merge, cv2.COLOR_HSV2BGR)
            bgr_left = cv2.cvtColor(merge_left, cv2.COLOR_HSV2BGR)
            bgr_right = cv2.cvtColor(merge_right, cv2.COLOR_HSV2BGR)
            img = resize(bgr, (48, 64), mode='constant')
            img_left = resize(bgr_left, (48, 64), mode='constant')
            img_right = resize(bgr_right, (48, 64), mode='constant')
        
            if self.episode == 4000:
                global mean_mean
                mean_mean = sum(self.mean_training) / 4001
                print(mean_mean)

        if self.episode > 4000:
            img_hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
            img_hsv_left = cv2.cvtColor(self.cv_left_image, cv2.COLOR_BGR2HSV)
            img_hsv_right = cv2.cvtColor(self.cv_right_image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(img_hsv)
            h_left, s_left, v_left = cv2.split(img_hsv_left)
            h_right, s_right, v_right = cv2.split(img_hsv_right)
            mean = np.mean(v)
            mean_left = np.mean(v_left)
            mean_right = np.mean(v_right)

            if  abs(mean - mean_mean) <= 10:
                gamma = 1.0
                look_up_table = np.zeros((256, 1) ,dtype=np.uint8)
                for i in range(256):
                    look_up_table[i][0] = (i/255)**(1.0/gamma)*255
                v_lut = cv2.LUT(v, look_up_table)
                v_lut_left = cv2.LUT(v_left, look_up_table)
                v_lut_right = cv2.LUT(v_right, look_up_table)
                merge = cv2.merge([h, s, v_lut])
                merge_left = cv2.merge([h_left, s_left, v_lut_left])
                merge_right = cv2.merge([h_right, s_right, v_lut_right])
                # print(np.mean(v_lut))   
                bgr = cv2.cvtColor(merge, cv2.COLOR_HSV2BGR)
                bgr_left = cv2.cvtColor(merge_left, cv2.COLOR_HSV2BGR)
                bgr_right = cv2.cvtColor(merge_right, cv2.COLOR_HSV2BGR)
                img = resize(bgr, (48, 64), mode='constant')
                img_left = resize(bgr_left, (48, 64), mode='constant')
                img_right = resize(bgr_right, (48, 64), mode='constant')


            if  abs(mean - mean_mean) > 10 and mean < 50:
                x = self.numbers_1[self.i]
                gamma = x
                look_up_table = np.zeros((256, 1) ,dtype=np.uint8)
                for i in range(256):
                    look_up_table[i][0] = (i/255)**(1.0/gamma)*255
                if self.i < 10:
                    self.i = self.i + 1
                if self.i >= 10:
                    self.i = 0
                v_lut = cv2.LUT(v, look_up_table)
                v_lut_left = cv2.LUT(v_left, look_up_table)
                v_lut_right = cv2.LUT(v_right, look_up_table)
                merge = cv2.merge([h, s, v_lut])
                merge_left = cv2.merge([h_left, s_left, v_lut_left])
                merge_right = cv2.merge([h_right, s_right, v_lut_right])
                # print(np.mean(v_lut))   
                bgr = cv2.cvtColor(merge, cv2.COLOR_HSV2BGR)
                bgr_left = cv2.cvtColor(merge_left, cv2.COLOR_HSV2BGR)
                bgr_right = cv2.cvtColor(merge_right, cv2.COLOR_HSV2BGR)
                img = resize(bgr, (48, 64), mode='constant')
                img_left = resize(bgr_left, (48, 64), mode='constant')
                img_right = resize(bgr_right, (48, 64), mode='constant')


            if  abs(mean - mean_mean) > 10 and mean > 50 :
                x = self.numbers_2[self.i]
                gamma = x
                look_up_table = np.zeros((256, 1) ,dtype=np.uint8)
                for i in range(256):
                    look_up_table[i][0] = (i/255)**(1.0/gamma)*255
                    
                if self.i < 10:
                    self.i = self.i + 1
                if self.i >= 10:
                    self.i = 0

                v_lut = cv2.LUT(v, look_up_table)
                v_lut_left = cv2.LUT(v_left, look_up_table)
                v_lut_right = cv2.LUT(v_right, look_up_table)
                merge = cv2.merge([h, s, v_lut])
                merge_left = cv2.merge([h_left, s_left, v_lut_left])
                merge_right = cv2.merge([h_right, s_right, v_lut_right])

                # print(np.mean(v_lut))   
                bgr = cv2.cvtColor(merge, cv2.COLOR_HSV2BGR)
                bgr_left = cv2.cvtColor(merge_left, cv2.COLOR_HSV2BGR)
                bgr_right = cv2.cvtColor(merge_right, cv2.COLOR_HSV2BGR)
                img = resize(bgr, (48, 64), mode='constant')
                img_left = resize(bgr_left, (48, 64), mode='constant')
                img_right = resize(bgr_right, (48, 64), mode='constant')
            
            else :
                gamma = 1.0
                look_up_table = np.zeros((256, 1) ,dtype=np.uint8)
                for i in range(256):
                    look_up_table[i][0] = (i/255)**(1.0/gamma)*255

                v_lut = cv2.LUT(v, look_up_table)
                v_lut_left = cv2.LUT(v_left, look_up_table)
                v_lut_right = cv2.LUT(v_right, look_up_table)
                
                merge = cv2.merge([h, s, v_lut])
                merge_left = cv2.merge([h_left, s_left, v_lut_left])
                merge_right = cv2.merge([h_right, s_right, v_lut_right])

                # print(np.mean(v_lut))   
                
                bgr = cv2.cvtColor(merge, cv2.COLOR_HSV2BGR)
                bgr_left = cv2.cvtColor(merge_left, cv2.COLOR_HSV2BGR)
                bgr_right = cv2.cvtColor(merge_right, cv2.COLOR_HSV2BGR)
                
                img = resize(bgr, (48, 64), mode='constant')
                img_left = resize(bgr_left, (48, 64), mode='constant')
                img_right = resize(bgr_right, (48, 64), mode='constant')

            
            


        ros_time = str(rospy.Time.now())

        if self.episode == 4000:
            self.learning = False
            self.dl.save(self.save_path)
            # self.dl.load("/home/yuzuki/model_gpu.pt")

        if self.episode == 5700:
            os.system('killall roslaunch')
            sys.exit()

        if self.learning:
            target_action = self.action
            distance = self.min_distance

            if self.mode == "manual":
                if distance > 0.1:
                    self.select_dl = False
                elif distance < 0.05:
                    self.select_dl = True
                if self.select_dl and self.episode >= 0:
                    target_action = 0
                action, loss = self.dl.act_and_trains(img , target_action)
                if abs(target_action) < 0.1:
                    action_left,  loss_left  = self.dl.act_and_trains(img_left , target_action - 0.2)
                    action_right, loss_right = self.dl.act_and_trains(img_right , target_action + 0.2)
                angle_error = abs(action - target_action)

            elif self.mode == "zigzag":
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
                    target_action = 0

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
                    line = [str(self.episode), "training", str(distance), str(self.pos_x), str(self.pos_y), str(self.pos_the)  ]
                    with open(self.path + self.start_time + '/' + 'training.csv', 'a') as f:
                        writer = csv.writer(f, lineterminator='\n')
                        writer.writerow(line)
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
                    line = [str(self.episode), "training", str(distance), str(self.pos_x), str(self.pos_y), str(self.pos_the)  ]
                    with open(self.path + self.start_time + '/' + 'training.csv', 'a') as f:
                        writer = csv.writer(f, lineterminator='\n')
                        writer.writerow(line)
                    with open(self.path + self.start_time + '/' + 'training.csv', 'a') as f:
                        writer = csv.writer(f, lineterminator='\n')
                        writer.writerow(line)


                angle_error = abs(action - target_action)
                if distance > 0.1:
                    self.select_dl = False
                elif distance < 0.05:
                    self.select_dl = True
                if self.select_dl and self.episode >= 0:
                    target_action = action

            elif self.mode == "follow_line":
                action, loss = self.dl.act_and_trains(img , target_action)
                if abs(target_action) < 0.1:
                    action_left,  loss_left  = self.dl.act_and_trains(img_left , target_action - 0.2)
                    action_right, loss_right = self.dl.act_and_trains(img_right , target_action + 0.2)
                angle_error = abs(action - target_action)

            elif self.mode == "selected_training":
                action = self.dl.act(img )
                angle_error = abs(action - target_action)
                loss = 0
                if angle_error > 0.05:
                    action, loss = self.dl.act_and_trains(img , target_action)
                    if abs(target_action) < 0.1:
                        action_left,  loss_left  = self.dl.act_and_trains(img_left , target_action - 0.2)
                        action_right, loss_right = self.dl.act_and_trains(img_right , target_action + 0.2)
                
                if distance > 0.15 or angle_error > 0.3:
                    self.select_dl = False
                # if distance > 0.1:
                #     self.select_dl = False
                elif distance < 0.05:
                    self.select_dl = True
                if self.select_dl and self.episode >= 0:
                    target_action = action

            # end mode

            self.episode += 1
            print(str(self.episode) + ", training, loss: " + str(loss) + ", angle_error: " + str(angle_error) + ", distance: " + str(distance))
            # print(str(self.episode)  + ", distance: " + str(distance))
            # line = [str(self.episode), "training", str(loss), str(angle_error), str(distance), str(self.pos_x), str(self.pos_y), str(self.pos_the)  ]
            line = [str(self.episode), "training", str(distance), str(self.pos_x), str(self.pos_y), str(self.pos_the)  ]
            with open(self.path + self.start_time + '/' + 'training.csv', 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(line)
            self.vel.linear.x = 0.2
            self.vel.angular.z = target_action
            self.nav_pub.publish(self.vel)

        else:
            target_action = self.dl.act(img)
            distance = self.min_distance
            print(str(self.episode) + ", test, angular:" + str(target_action) + ", distance: " + str(distance))

            self.episode += 1
            angle_error = abs(self.action - target_action)
            # line = [str(self.episode), "test", "0", str(angle_error), str(distance), str(self.pos_x), str(self.pos_y), str(self.pos_the)  ]
            line = [str(self.episode), "test", str(distance), str(self.pos_x), str(self.pos_y), str(self.pos_the)  ]
            with open(self.path + self.start_time + '/' + 'training.csv', 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(line)
            self.vel.linear.x = 0.2
            self.vel.angular.z = target_action
            self.nav_pub.publish(self.vel)

        temp = copy.deepcopy(bgr)
        cv2.imshow("HSV Center Image", temp)
        temp = copy.deepcopy(bgr_left)
        cv2.imshow("HSV Left Image", temp)
        temp = copy.deepcopy(bgr_right)
        cv2.imshow("HSV Right Image", temp)
        cv2.imshow("/camera/rgb/image_raw", self.cv_image)
        cv2.waitKey(1)

if __name__ == '__main__':
    rg = nav_cloning_node()
    DURATION = 0.2
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()