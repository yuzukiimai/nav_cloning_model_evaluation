#!/usr/bin/env python3
import glob
import os

lists = []

for filename in sorted(glob.glob('/home/yuzuki/test1_ws/src/nav_cloning/data/result_change_dataset_balance/*/episode.csv'), key=lambda f: os.stat(f).st_mtime, reverse=True):
# for filename in sorted(glob.glob('/home/yuzuki/test1_ws/src/nav_cloning/data/result_use_dl_output/*/episode.csv'), key=lambda f: os.stat(f).st_mtime, reverse=True):
    lists.append(filename)

def draw_training_pos():
    index = 0
    file_number = 1
    while index < 50:
        print("--------------------------------")
        print(file_number)
        with open(lists[index]) as f:
            print(f.read())

        index += 1
        file_number += 1

if __name__ == '__main__':
    draw_training_pos()

