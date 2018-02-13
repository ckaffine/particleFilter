import pdb
import numpy as np

filename = "../data/log/robotdata1.log"

prev_pos = None

eq_values = {}
eq_counts = {}
with open(filename, 'r') as f:
    for i, line in enumerate(f):
        if line[0] != "L": continue

        split = line.split()
        curr_pos = split[1:4]
        curr_pos = map(float, curr_pos)

        key = tuple(curr_pos)
        lidar = map(float, split[7:-1])

        eq_values[key] = eq_values.get(key, []) + [lidar]
        eq_counts[key] = eq_counts.get(key, 0) + 1

        # if i > 10:
            # break

filtered = {}
for key in eq_values:
    if eq_counts[key] > 5:
        filtered[key] = np.array(eq_values[key])

# Each value in filtered is [M x 180], some number of sensor readings right on
# top of each other. Plot the difference from the mean for each of the
# measurements.

from matplotlib import pyplot as plt
