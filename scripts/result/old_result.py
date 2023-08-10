#!/usr/bin/env python3
import csv
from matplotlib import pyplot
from matplotlib.patches import Circle
import numpy as np
from PIL import Image


def draw_training_pos():
    image = Image.open(('/home/yuzuki/catkin_ws/src/nav_cloning')+'/maps/2_3_result.png').convert("L")
    arr = np.asarray(image)
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.imshow(arr, cmap='gray', extent=[-15,11,-11,37.2])
    vel = 0.2
    arrow_dict = dict(arrowstyle = "->", color = "black")
    count = 0
    with open('/home/yuzuki/catkin_ws/src/nav_cloning/data/result_change_dataset_balance/20230728_15:18:04/training_all.csv', 'r') as f:
    
        for row in csv.reader(f):
                str_step, mode,distance,str_x, str_y, str_the = row
                if mode == "training":
                    x, y = float(str_x), float(str_y)
                    patch = Circle(xy=(x, y), radius=0.08, facecolor="red") 
                    ax.add_patch(patch)
        else:
                    pass
            
    ax.set_xlim([-13, 10])
    ax.set_ylim([-10, 35])
    # pyplot.savefig("/home/yuzuki/research/result_gray_0510_2/png/1.0.png") 
    pyplot.show()

if __name__ == '__main__':
    draw_training_pos()
