import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.stats import expon
import pdb
import cv2

from MapReader import MapReader


def cached_x(f):
    cache = {}
    def wrapper(self, x): # [cm, cm, rad]
        key = (math.floor(x[0] / 10), math.floor(x[1] / 10),
                math.floor(x[2] / math.radians(5)))
        if key not in cache:
            cache[key] = f(self, x)
        return cache[key]

    return wrapper


class SensorModel:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    z_max = 8183
    # thresh = 0.01 # Confidence threshold for raytracing.
    thresh = 0.3 # Confidence threshold for raytracing.
    close_thresh = 3

    # # Parameters for the linear combination of errors, also to tune.
    # # We know that these have worked before.
    # stdev = 100 # Gaussian
    # scale = 1000
    # k_hit = 1.0 / norm.pdf(0, 0, stdev)
    # k_short = 100
    # k_miss = 50
    # k_rand = 2500

    # Error parameters to tune.
    stdev = 100 # Gaussian
    scale = 1000

    # Parameters
    k_hit = 1.0 / norm.pdf(0, 0, stdev)
    k_short = 50
    k_miss = 10
    k_rand = 2500

    def __init__(self, occupancy_map):

        """
        TODO : Initialize Sensor Model parameters here
        """
        self.map = occupancy_map > self.thresh

        # This will be unhappy if we give it a bad map.
        self.max_y, self.max_x = self.map.shape
        self.max_y -= 1
        self.max_x -= 1
        self.min_y, self.min_x = 0, 0

        self.lut = np.load('raycast.npy')


    def beam_range_measurement(self, x_t1):
        """
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] z_t1* : actual measurement according to map
        """
        # Given an [x, y, theta] pair, walk down theta until an obstacle is
        # found. This will also do the proper unit conversions - the x, y
        # positions are given in raw centimeters, while the map is given in 10cm
        # blocks.

        step = 0.5 # tiles
        x_t1 *= np.array((.1, .1, 180 / math.pi))
        x_t1 = np.floor(x_t1).astype(np.int32)
        return self.lut[x_t1[:, 0], x_t1[:, 1], x_t1[:, 2]]

#        x = x_t1[0] / 10.0
#        y = x_t1[1] / 10.0
#        th = x_t1[2]

        #return self.lut[int(x), int(y), int(th * 180 / math.pi)]

        """
        dx = step * math.cos(th)
        dy = step * math.sin(th)

        while True:

            # If the position is outside the boundaries of the map, just return
            # max distance for now.
            tx = int(math.floor(x))
            ty = int(math.floor(y))

            # Going out of bounds of the map
            if (tx > self.max_x or tx < self.min_x or
                    ty > self.max_y or ty < self.min_y):
                return self.z_max

            # Within bounds, check if something's there.
            curr = self.map[ty][tx]
            if curr: # detected something
                dist = math.sqrt((x * 10 - x_t1[0]) ** 2 +
                                 (y * 10 - x_t1[1]) ** 2)
                return min(dist, self.z_max)

            x += dx
            y += dy
        """


    def _hit_error(self, x, x_star):
        # We should never get a value that's outside the bounds of the possible
        # sensor measurements, so there's no reason to case on returning 0.
        p = norm.pdf(x, x_star, self.stdev)
        n = 1.0 / (norm.cdf(self.z_max, x_star, self.stdev) -
                   norm.cdf(0, x_star, self.stdev))
        return n * p


    def _short_error(self, x, x_star):
        p = expon.pdf(x, 0, self.scale)
        n = 1.0 / (norm.cdf(x_star, 0, self.scale))
        return n * p


    def _miss_error(self, x, x_star):
        return (np.array(x) == self.z_max).astype(int)


    def _rand_error(self, x, x_star):
        return 1.0 / self.z_max


    def _total_error(self, x, x_star):
        e_hit = self._hit_error(x, x_star)
        e_short = self._short_error(x, x_star)
        e_miss = self._miss_error(x, x_star)
        e_rand = self._rand_error(x, x_star)

        return (self.k_hit * e_hit +
                self.k_short * e_short +
                self.k_miss * e_miss +
                self.k_rand * e_rand)


    def _close_counts(self, x, x_star):
        return (np.abs(x - x_star) < self.close_thresh).sum()


    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """

        th = x_t1[2]

        # Calculate where the laser is
        laser_offset = 25 # cm
        laser_dx = laser_offset * math.cos(th)
        laser_dy = laser_offset * math.sin(th)
        laser_x = x_t1[0] + laser_dx
        laser_y = x_t1[1] + laser_dy

        num = 36
        z_stars = np.zeros(num)
        t = np.linspace(th - math.pi / 2,
                        th + math.pi / 2,
                        num=num)
        pos = np.vstack((laser_x*np.ones((num)), laser_y*np.ones((num)), t)).T
        z_stars = self.beam_range_measurement(pos)

        i_to_use = np.linspace(0, len(z_t1_arr), endpoint=False, num=num)
        i_to_use = i_to_use.astype(int)
        z_to_use = z_t1_arr[i_to_use]

        p = self._total_error(z_to_use, z_stars)
        p = np.exp(np.sum(np.log(p)))
        # p = self._close_counts(z_to_use, z_stars)
        return p


def test_beam_range_measurement():
    src_path_map = '../data/map/wean.dat'
    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    model = SensorModel(occupancy_map)

    fig = plt.figure()
    mng = plt.get_current_fig_manager();  # mng.resize(*mng.window.maxsize())
    plt.imshow(occupancy_map, cmap='Greys')

    # x = 4650.0
    # y = 2200.0

    # x = 6000.0
    # y = 1450.0

    # x = 1660.0
    # y = 6100.0

    # x = 4833
    # y = 5111

    while True:

        while True:
            y = np.random.uniform(0, 7000)
            x = np.random.uniform(3000, 7000)
            if occupancy_map[int(y / 10)][int(x / 10)] == 0:
                break

        plt.clf()
        plt.imshow(occupancy_map, cmap='Greys')

        test_points = np.zeros((360, 3))
        for deg in np.linspace(0, 360, num=360, endpoint=False):
            th = math.radians(deg)
            test_points[np.floor(deg).astype(np.uint32), :] = (np.array([x, y, th]))
        dists = model.beam_range_measurement(test_points)
        for deg in range(360):
            th = math.radians(deg)
            dist = dists[deg]
            print "Distance of", dist, "for", deg, "degrees"

            x2 = x + dist * math.cos(th)
            y2 = y + dist * math.sin(th)
            plt.plot([x / 10, x2 / 10], [y / 10, y2 / 10], 'b-', linewidth=0.25)

        plt.plot([x / 10], [y / 10], marker='o', markersize=3, color="red")
        plt.axis([0, 800, 0, 800])
        plt.pause(1)

    # plt.show()


def test_error():
    src_path_map = '../data/map/wean.dat'
    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    model = SensorModel(occupancy_map)

    x = np.linspace(0, 8183, num=2500)
    y = model._total_error(x, np.ones(x.shape) * 2500)
    plt.plot(x, y)
    plt.axis([-100, 8283, 0, 1.5])
    plt.show()




if __name__=='__main__':
    # This conducts a bunch of tests on the Sensor Model
    # test_beam_range_measurement()
    test_error()
