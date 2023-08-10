#!/usr/bin/env python3
import csv
from matplotlib import pyplot
from matplotlib.patches import Circle
import numpy as np
from PIL import Image
import glob




def draw_training_pos_willow():
    image = Image.open("/home/yuzuki/catkin_ws/src/nav_cloning/maps/map.png").convert("L")
    arr = np.asarray(image)
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.imshow(arr, cmap='gray', extent=[-10,50,-10,50])
    vel = 0.2
    arrow_dict = dict(arrowstyle = "->", color = "black")
    count = 0
    for filename in glob.glob('/home/yuzuki/catkin_ws/src/nav_cloning/data/result_change_dataset_balance/*/'):
        with open(filename + 'training_all.csv', 'r') as f:
            for row in csv.reader(f):
                    str_step, mode,distance,str_x, str_y, str_the = row
                    if mode == "test":
                        x, y = float(str_x), float(str_y)
                        patch = Circle(xy=(x, y), radius=0.05, facecolor="red") 
                        ax.add_patch(patch)
            else:
                        pass
            
    ax.set_xlim([-5, 30])
    ax.set_ylim([-5, 15])
    pyplot.show()





def draw_training_pos_tsudanuma_real():
    image = Image.open(('/home/yuzuki/imai_gamma_ws/src/nav_cloning')+'/maps/2_3_result.png').convert("L")
    arr = np.asarray(image)
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.imshow(arr, cmap='gray', extent=[-9.5,38.5,-8.5,17])
    vel = 0.2
    arrow_dict = dict(arrowstyle = "->", color = "black")
    count = 0
    for filename in glob.glob('/home/yuzuki/catkin_ws/src/nav_cloning/data/result_change_dataset_balance/*/'):
        with open(filename + 'training_all.csv', 'r') as f:
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
    pyplot.show()






def draw_training_pos_tsudanuma():
    image = Image.open(('/home/yuzuki/catkin_ws/src/nav_cloning')+'/maps/2_3_result.png').convert("L")
    arr = np.asarray(image)
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.imshow(arr, cmap='gray', extent=[-15,11,-11,37.2])
    vel = 0.2
    arrow_dict = dict(arrowstyle = "->", color = "black")
    count = 0
    for filename in glob.glob('/home/yuzuki/catkin_ws/src/nav_cloning/data/result_change_dataset_balances/*/'):
        with open(filename + 'training_all.csv', 'r') as f:
            for row in csv.reader(f):
                    str_step, mode,distance,str_x, str_y, str_the = row
                    if mode == "test":
                        x, y = float(str_x), float(str_y)
                        patch = Circle(xy=(x, y), radius=0.02, facecolor="red") 
                        ax.add_patch(patch)
            else:
                        pass
            
    ax.set_xlim([-13, 10])
    ax.set_ylim([-10, 35]) 
    pyplot.show()




if __name__ == '__main__':
    draw_training_pos_willow()
    # draw_training_pos_tsudanuma_real()
    # draw_training_pos_tsudanuma()


