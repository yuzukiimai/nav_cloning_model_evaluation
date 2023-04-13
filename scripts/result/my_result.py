#!/usr/bin/env python3
from __future__ import print_function
# import roslib
# roslib.load_manifest('nav_cloning')
# import rospy
import csv
import math
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Polygon
import numpy as np
from PIL import Image
import glob
import os

lists = []

for filename in sorted(glob.glob('/home/yuzuki/test1_ws/src/nav_cloning/data/result_change_dataset_balance/*/training.csv'), key=lambda f: os.stat(f).st_mtime, reverse=True):
# for filename in sorted(glob.glob('/home/yuzuki/test1_ws/src/nav_cloning/data/result_use_dl_output/*/training.csv'), key=lambda f: os.stat(f).st_mtime, reverse=True):
    lists.append(filename)

def draw_training_pos():
    index = 0
    file_number = 1
    while index < 50:
        # rospy.init_node('draw_training_pos_node', anonymous=True)
        image = Image.open(('/home/yuzuki/catkin_ws/src/nav_cloning')+'/maps/map.png').convert("L")
        arr = np.asarray(image)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(arr, cmap='gray', extent=[-10,50,-10,50])
        vel = 0.2
        arrow_dict = dict(arrowstyle = "->", color = "black")
        count = 0
        
        print(file_number)
        with open(lists[index]) as f:
            for row in csv.reader(f):
                    str_step, mode,distance,str_x, str_y, str_the = row
                    if mode == "test":
                        x, y = float(str_x), float(str_y)
                        patch = Circle(xy=(x, y), radius=0.08, facecolor="red") 
                        ax.add_patch(patch)
            else:
                        pass
                
        ax.set_xlim([-5, 30])
        ax.set_ylim([-5, 15])
        plt.show(block=False)
        plt.pause(3)
        plt.close()

        index += 1
        file_number += 1

if __name__ == '__main__':
    draw_training_pos()

