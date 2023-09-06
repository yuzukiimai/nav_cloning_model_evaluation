#!/usr/bin/env python3
import glob
import os

lists = []
for filename in sorted(glob.glob('/home/yuzuki/catkin_ws/src/nav_cloning/data/result_change_dataset_balance/*/learning_exit_episode.csv'), key=lambda f: os.stat(f).st_mtime, reverse=True):
# for filename in sorted(glob.glob('/home/yuzuki/result/result_SI/50%/*/learning_exit_episode.csv'), key=lambda f: os.stat(f).st_mtime, reverse=True):
    lists.append(filename)

# def draw_training_pos():
#     index = 0
#     file_number = 1
#     while index < 100:
#         print("--------------------------------")
#         print(file_number)
#         with open(lists[index]) as f:
#             print(f.read())

#         index += 1
#         file_number += 1

def draw_training_pos():
    index = 0
    file_number = 1
    sum_values = 0
    num_values = 0

    while index < 100:
        print("--------------------------------")
        print(file_number)
        with open(lists[index]) as f:
            data = f.read()

        # Assuming the CSV file contains numerical values in each line, one per line.
        values = [float(line.strip()) for line in data.split('\n') if line.strip()]

        # Calculate sum and count
        sum_values += sum(values)
        num_values += len(values)

        # Print the contents of the current file
        print(data)

        index += 1
        file_number += 1

    # Calculate the average
    if num_values > 0:
        average = sum_values / num_values
        print("-------------------------------")
        print(f"Total: {sum_values}")
        print(f"Average: {average}")



if __name__ == '__main__':
    draw_training_pos()

