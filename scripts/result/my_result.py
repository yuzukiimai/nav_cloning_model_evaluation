#!/usr/bin/env python3
import csv
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from PIL import Image
import glob
import os

lists = []

for filename in sorted(glob.glob('/home/yuzuki/catkin_ws/src/nav_cloning/data/result_change_dataset_balance/*/training_all.csv'), key=lambda f: os.stat(f).st_mtime, reverse=True):
    lists.append(filename)

def draw_training_pos_willow():
    index = 0
    file_number = 1
    while index < 100:
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
        # plt.savefig(str(file_number) + ".png") 
        plt.pause(1)
        plt.close()

        index += 1
        file_number += 1


def draw_training_pos_tsudanuma_real():
    index = 0
    file_number = 1
    while index < 50:
        image = Image.open(('/home/yuzuki/catkin_ws/src/nav_cloning')+'/maps/2_3_result.png').convert("L")
        arr = np.asarray(image)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(arr, cmap='gray', extent=[-9.5,38.5,-8.5,17])
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
                
        ax.set_xlim([-8, 37])
        ax.set_ylim([-5, 15])
        plt.show(block=False)
        # plt.savefig(str(file_number) + ".png") 
        plt.pause(3)
        plt.close()

        index += 1
        file_number += 1



def draw_training_pos_tsudanuma():
    index = 0
    file_number = 1
    while index < 50:
        image = Image.open(('/home/yuzuki/catkin_ws/src/nav_cloning')+'/maps/2_3_result.png').convert("L")
        arr = np.asarray(image)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(arr, cmap='gray', extent=[-15,11,-11,37.2])
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

            
        ax.set_xlim([-13, 10])
        ax.set_ylim([-10, 35]) 
        plt.show(block=False)
        # plt.savefig(str(file_number) + ".png") 
        plt.pause(3)
        plt.close()

        index += 1
        file_number += 1


if __name__ == '__main__':
    # draw_training_pos_willow()
    # draw_training_pos_tsudanuma_real()
    draw_training_pos_tsudanuma()


