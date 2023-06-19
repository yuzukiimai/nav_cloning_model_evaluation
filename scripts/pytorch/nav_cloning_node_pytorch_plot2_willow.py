#!/usr/bin/env python3
import roslib
roslib.load_manifest('nav_cloning')
import rospy
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from PIL import Image
from matplotlib.patches import Circle
import csv
from std_srvs.srv import SetBool, SetBoolResponse
from std_msgs.msg import Bool


class nav_cloning_node2:
    def __init__(self):
        rospy.init_node('nav_cloning_node2', anonymous=True)
        self.srv = rospy.Service('/switch_segmentation', SetBool, self.callback_dl_switch)
        self.exit_pub = rospy.Publisher("/learning_exit", Bool, queue_size=1)
        self.num = 0
        self.i = 0
        self.exit_flg = False
        self.loop_flg = False
        self.laps = False
        self.loop_count = 0

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim([-5, 30])
        self.ax.set_ylim([-5, 15])
        self.ax.text(15, -4.8, "Total: " + str(self.num), color="b", fontsize="xx-large")
        self.ax.text(-4.2, -4.8, "Number of laps: " + str(self.i), color="r", fontsize="xx-large")
        self.image = Image.open('/home/yuzuki/catkin_ws/src/nav_cloning/maps/map.png').convert("L")
        self.arr = np.asarray(self.image)
        self.map_plot = self.ax.imshow(self.arr, cmap='gray', extent=[-10,50,-10,50])
        self.patches = [] 
    
    def callback_dl_switch(self, data):
        resp = SetBoolResponse()
        self.loop_flg = data.data
        if self.loop_flg and self.loop_count == 0:
            self.laps = True
        if self.loop_flg == False:
            self.loop_count = 0
        resp.message = "switch"
        resp.success = True
        return resp

    def update(self, frame):
        self.ax.clear()
        self.ax.set_xlim([-5, 30])
        self.ax.set_ylim([-5, 15])
        self.ax.text(15, -4.8, "Total: " + str(self.num), color="b", fontsize="xx-large")
        self.ax.text(-4.2, -4.8, "Number of laps: " + str(self.i), color="r", fontsize="xx-large")
        self.map_plot = self.ax.imshow(self.arr, cmap='gray', extent=[-10,50,-10,50])

        # with open('/home/yuzuki/catkin_ws/src/nav_cloning/data/result_follow_line/training.csv', 'r') as f:
        with open('/home/yuzuki/catkin_ws/src/nav_cloning/data/result_change_dataset_balance/training.csv', 'r') as f:
            for row in csv.reader(f):
                str_step, mode, distance, str_x, str_y, str_the = row
                # if mode == "training":
                x, y = float(str_x), float(str_y)
                self.patches.append(Circle(xy=(x, y), radius=0.10, facecolor="b"))
                self.num += 1
        
        # with open('/home/yuzuki/catkin_ws/src/nav_cloning/data/result_follow_line/training.csv', 'w') as f:
        with open('/home/yuzuki/catkin_ws/src/nav_cloning/data/result_change_dataset_balance/training.csv', 'w') as f:
            f.write("")
                    
        for patch in self.patches:
            self.ax.add_patch(patch)
            

        if self.num > 800:
            self.exit_flg = True
            self.exit_pub.publish(self.exit_flg)

        if self.laps:
            # plt.savefig("/home/yuzuki/catkin_ws/src/nav_cloning/data/result_follow_line/" + str(self.i) + ".png") 
            plt.savefig("/home/yuzuki/catkin_ws/src/nav_cloning/data/result_change_dataset_balance/" + str(self.i) + ".png") 
            self.loop_count = 1
            self.laps = False
            self.i += 1
            self.num = 0
            self.patches.clear()
            
            

       

    def animate(self):
        animation = FuncAnimation(self.fig, self.update, frames=10, interval=3000)
        plt.show()


if __name__ == '__main__':
    # with open('/home/yuzuki/catkin_ws/src/nav_cloning/data/result_follow_line/training.csv', 'w') as f:
    with open('/home/yuzuki/catkin_ws/src/nav_cloning/data/result_change_dataset_balance/training.csv', 'w') as f:
            f.write("")
    l = nav_cloning_node2()
    l.animate()