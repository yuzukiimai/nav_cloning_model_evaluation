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
            x = round(random.uniform(0.8, 2.0), 1)
            gamma = x
            look_up_table = np.zeros((256, 1) ,dtype=np.uint8)
            for i in range(256):
                look_up_table[i][0] = (i/255)**(1.0/gamma)*255

            img_hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(img_hsv)  
            v_lut = cv2.LUT(v, look_up_table) 
            s_lut = cv2.LUT(s, look_up_table)
            merge = cv2.merge([h, s_lut, v_lut]) 
            bgr = cv2.cvtColor(merge, cv2.COLOR_HSV2BGR)
            
            print(bgr)

            temp = copy.deepcopy(bgr)
            cv2.imshow("gamma_Image", temp)
            cv2.waitKey(1)

if __name__ == '__main__':
    rg = img_node()
    DURATION = 0.2
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()