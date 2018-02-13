import numpy as np
import sys
import pdb
import cv2
import math
import time
import os

from MapReader import MapReader
from MotionModel import MotionModel
from SensorModel import SensorModel
from Resampling import Resampling
from Visualizer import Visualizer

def init_particles_freespace(num_particles, occupancy_map, num_angles=6):
    # initialize [x, y, theta] positions in world_frame for all particles
    # (randomly across the map)
    min_y = 3200
    max_y = 4500
    min_x = 3400
    max_x = 5500

    min_x = 3000
    max_x = 7000
    min_y = 0
    max_y = 7500

    num_positions = num_particles / num_angles

    y0_vals = np.random.uniform( min_y, max_y, (num_positions, 1) )
    x0_vals = np.random.uniform( min_x, max_x, (num_positions, 1) )

    X_bar_init = []

    # Force the points to be in the map
    for i in range(num_positions):
        x = x0_vals[i]
        y = y0_vals[i]

        while True:
            curr = occupancy_map[int(math.floor(y / 10))][int(math.floor(x / 10))]

            # Resample
            if curr == 0:
                break
            else:
                y = np.random.uniform(min_y, max_y)
                x = np.random.uniform(min_x, max_x)

        # x0_vals[i] = x
        # y0_vals[i] = y
        offset_range = 360 / num_angles
        offset = np.random.randint(-offset_range / 2, offset_range / 2)
        for i in range(num_angles):
            th = offset + offset_range * i
            th = math.radians(th)
            th = (th + math.pi) % (2 * math.pi) - math.pi
            X_bar_init.append([x, y, th, 1])

    X_bar_init = np.array(X_bar_init)
    return X_bar_init


def main(src_path_log='../data/log/robotdata1.log', num_particles=1000):
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """
    print "Executing from", src_path_log, "with", num_particles, "particles."

    src_path_map = '../data/map/wean.dat'

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    cv2.imwrite("test.jpg", occupancy_map * 255)
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel(occupancy_map)
    sensor_model = SensorModel(occupancy_map)
    resampler = Resampling()

    X_bar = init_particles_freespace(num_particles, occupancy_map, 5)

    # Saving the visualization
    basename = os.path.basename(src_path_log)
    basename = os.path.splitext(basename)[0]
    vis = Visualizer(skips=10, header=basename + "_" + "%05d" % num_particles)

    # Localization Main Loop
    first_time_idx = True
    for time_idx, line in enumerate(logfile):

        # Move the visualization up to the top, so it visualizes every frame.
        vis.visualize(occupancy_map, X_bar, time_idx)
        vis.save()

        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        meas_type = line[0] # L : laser scan measurement, O : odometry measurement
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ') # convert measurement values from string to double

        odometry_robot = meas_vals[0:3] # odometry reading [x, y, theta] in odometry frame
        time_stamp = meas_vals[-1]

        # If this is commented, we're keeping all measurements.
        # if ((time_stamp <= 0.0) | (meas_type == "O")): # ignore pure odometry measurements for now (faster debugging)
            # continue

        if (meas_type == "L"):
             odometry_laser = meas_vals[3:6] # [x, y, theta] coordinates of laser in odometry frame
             ranges = meas_vals[6:-1] # 180 range measurement values from single laser scan
             # visualize_laser(odometry_laser, ranges)

        print "Processing time step " + str(time_idx) + " at time " + str(time_stamp) + "s"

        if (first_time_idx):
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        X_bar_new = np.zeros((num_particles, 4), dtype=np.float64)
        u_t1 = odometry_robot

        # If we're in the same location as before, don't do anything.
        if (u_t0 == u_t1).all():
            print "Skipping all updates because we haven't moved"
            continue

        X_tmp = motion_model.update(u_t0, u_t1, X_bar)

        for m in range(0, num_particles):

            # ============ Motion Model ==============
            x_t1 = X_tmp[m, 0:3]
            #x_t1 = motion_model.update(u_t0, u_t1, x_t0)

            # ============ Sensor Model ==============
            if (meas_type == "L"):
                z_t = ranges
                w_t = sensor_model.beam_range_finder_model(z_t, x_t1)
                X_bar_new[m,:] = np.hstack((x_t1, w_t))
            else:
                X_bar_new[m,:] = np.hstack((x_t1, X_bar[m,3]))

        X_bar = X_bar_new
        u_t0 = u_t1

        # =========== Resampling ============
        X_bar = resampler.low_variance_sampler(X_bar)


    # Finish writing the movie
    vis.visualize(occupancy_map, X_bar, time_idx)
    vis.save()
    vis.close()

if __name__=="__main__":
    if len(sys.argv) >= 3:
        path = sys.argv[1]
        particle_count = int(sys.argv[2])
        main(path, particle_count)
    elif len(sys.argv) >= 2:
        path = sys.argv[1]
        main(path)
    else:
        main()

