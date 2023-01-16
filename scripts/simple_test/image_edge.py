#!/usr/bin/env python3
from numpy import dtype
import numpy as np
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import random
import copy
from skimage.transform import resize

class img_node:
    def __init__(self):
        rospy.init_node('img_node', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.callback)
        self.cv_image = np.zeros((480,640,3), np.uint8)
        
    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)


    def loop(self):
        if self.cv_image.size == 640 * 480 * 3:
            x = round(random.uniform(0.5, 1.2), 1)
            # y = round(random.uniform(0.5, 1.2), 1)
            # z = round(random.uniform(0.5, 1.2), 1)
            img_hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
            h_deg = 1
            s_mag = 1
            v_mag = x
            img_hsv[:,:,(0)] = img_hsv[:,:,(0)]+h_deg
            img_hsv[:,:,(1)] = img_hsv[:,:,(1)]*s_mag
            img_hsv[:,:,(2)] = img_hsv[:,:,(2)]*v_mag
            bgr = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)
            img = resize(bgr, (48, 64), mode='constant')
            print(bgr)

            temp = copy.deepcopy(bgr)
            # temp = copy.deepcopy(img)
            cv2.imshow("hsv_Image", temp)
            # cv2.imshow("Resized_hsv_Image", temp)
            cv2.waitKey(1)

if __name__ == '__main__':
    rg = img_node()
    DURATION = 0.2
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()