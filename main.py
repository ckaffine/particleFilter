import numpy as np
import sys
import pdb
import cv2
import math

from MapReader import MapReader
from MotionModel import MotionModel
from SensorModel import SensorModel
from Resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time

def visualize_map(occupancy_map):
    fig = plt.figure()
    # plt.switch_backend('TkAgg')
    mng = plt.get_current_fig_manager();  # mng.resize(*mng.window.maxsize())
    plt.ion(); plt.imshow(occupancy_map, cmap='Greys'); plt.axis([0, 800, 0, 800]);
    plt.pause(0.00001)


def visualize_timestep(X_bar, tstep):
    x_locs = X_bar[:,0]/10.0
    y_locs = X_bar[:,1]/10.0
    scat_back = plt.scatter(x_locs, y_locs, c='b', marker='o', s=5)
    plt.pause(0.00001)
    scat_back.remove()


def init_particles_freespace(num_particles, occupancy_map):
    # initialize [x, y, theta] positions in world_frame for all particles
    # (randomly across the map) 
    # min_y = 3200
    # max_y = 4500
    # min_x = 3400
    # max_x = 5500

    min_x = 3000
    max_x = 7000
    min_y = 0
    max_y = 7500

    y0_vals = np.random.uniform( min_y, max_y, (num_particles, 1) )
    x0_vals = np.random.uniform( min_x, max_x, (num_particles, 1) )
    theta0_vals = np.random.uniform( -math.pi, math.pi, (num_particles, 1) )

    for i in range(num_particles):
        x = x0_vals[i]
        y = y0_vals[i]

        while True:
            curr = occupancy_map[math.floor(y / 10)][math.floor(x / 10)]

            # Resample
            if curr == 0:
                break
            else:
                y = np.random.uniform(min_y, max_y)
                x = np.random.uniform(min_x, max_x)

        x0_vals[i] = x
        y0_vals[i] = y

    # Initialize weights for all particles to 1, since it doesn't matter.
    w0_vals = np.ones( (num_particles,1), dtype=np.float64)
    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init


def main():

    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]   
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """

    """
    Initialize Parameters
    """
    src_path_map = '../data/map/wean.dat'
    src_path_log = '../data/log/robotdata1.log'

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map() 
    cv2.imwrite("test.jpg", occupancy_map * 255)
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel(occupancy_map)
    sensor_model = SensorModel(occupancy_map)
    resampler = Resampling()

    num_particles = 500
    X_bar = init_particles_freespace(num_particles, occupancy_map)

    vis_flag = 1

    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    if vis_flag:
        visualize_map(occupancy_map)
        visualize_timestep(X_bar, 0)

    first_time_idx = True
    for time_idx, line in enumerate(logfile):

        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        meas_type = line[0] # L : laser scan measurement, O : odometry measurement
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ') # convert measurement values from string to double

        odometry_robot = meas_vals[0:3] # odometry reading [x, y, theta] in odometry frame
        time_stamp = meas_vals[-1]

        # If this is commented, we're keeping all measurements.
        # if ((time_stamp <= 0.0) | (meas_type == "O")): # ignore pure odometry measurements for now (faster debugging) 
            # continue

        if (meas_type == "L"):
             # odometry_laser = meas_vals[3:6] # [x, y, theta] coordinates of laser in odometry frame
             ranges = meas_vals[6:-1] # 180 range measurement values from single laser scan
        
        print "Processing time step " + str(time_idx) + " at time " + str(time_stamp) + "s"

        if (first_time_idx):
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        X_bar_new = np.zeros( (num_particles,4), dtype=np.float64)
        u_t1 = odometry_robot

        # If we're in the same location as before, don't do anything.
        if (u_t0 == u_t1).all():
            print "Skipping all updates because we haven't moved"
            continue

        for m in range(0, num_particles):

            """
            MOTION MODEL
            """
            x_t0 = X_bar[m, 0:3]
            x_t1 = motion_model.update(u_t0, u_t1, x_t0)

            """
            SENSOR MODEL
            """
            if (meas_type == "L"):
                z_t = ranges
                w_t = sensor_model.beam_range_finder_model(z_t, x_t1)
                X_bar_new[m,:] = np.hstack((x_t1, w_t))
            else:
                X_bar_new[m,:] = np.hstack((x_t1, X_bar[m,3]))
        
        X_bar = X_bar_new
        u_t0 = u_t1

        """
        RESAMPLING
        """
        X_bar = resampler.low_variance_sampler(X_bar)

        if vis_flag and time_idx % 5 == 0:
            visualize_timestep(X_bar, time_idx)


if __name__=="__main__":
    main()
