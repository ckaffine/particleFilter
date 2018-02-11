import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.stats import expon
import pdb
import cv2

from MapReader import MapReader

class SensorModel:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    z_max = 8183
    thresh = 0.0 # Confidence threshold for raytracing.

    # Error parameters to tune.
    stdev = 100 # Gaussian
    scale = 1000

    # Parameters for the linear combination of errors, also to tune.
    k_hit = 1.0 / norm.pdf(0, 0, stdev)
    # k_short = 50
    # k_miss = 1
    # k_rand = 100
    k_short = 0
    k_miss = 1
    k_rand = 0


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



    def beam_range_measurement(self, x_t1):
        """
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] z_t1* : actual measurement according to map
        """
        # Given an [x, y, theta] pair, walk down theta until an obstacle is
        # found. This will also do the proper unit conversions - the x, y
        # positions are given in raw centimeters, while the map is given in 10cm
        # blocks.

        step = 0.5

        x = x_t1[0] / 10.0
        y = x_t1[1] / 10.0
        th = x_t1[2]

        dx = step * math.cos(th)
        dy = step * math.sin(th)

        while True:

            # If the position is outside the boundaries of the map, just return
            # max distance for now.
            tx = int(math.floor(x))
            ty = int(math.floor(y))

            # Going out of bounds
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


    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """

        """
        TODO : Add your code here
        """
        p = 0 # Log probability of getting this measurement
       
        th = x_t1[2]
        # Calculate where the laser is
        laser_offset = 25 # cm
        laser_dx = laser_offset * math.cos(th)
        laser_dy = laser_offset * math.sin(th)

        laser_x = x_t1[0] + laser_dx
        laser_y = x_t1[1] + laser_dy

        z_stars = []

        # Annoyingly, there are 180 measurements across 180 degrees, not 181.
        # We'll reduce the number of measurements we take from the LIDAR from
        # using all 180, to some variable number.
        num_to_use = 180
        num_to_use = min(num_to_use, len(z_t1_arr))
        num_increments = num_to_use - 1
        inc = 180.0 / num_increments
        for i in range(num_to_use):
            offset = math.radians(inc * i - 90)
            pos = np.array([laser_x, laser_y, th + offset])
            z_star = self.beam_range_measurement(pos)
            z_stars.append(z_star)

        i_to_use = [i * (len(z_t1_arr) - 1) / num_increments
                    for i in range(num_to_use)]
        z_to_use = z_t1_arr[i_to_use]

        # Everything up to here should be safe and never throw an error, since
        # we're not doing anything crazy with small numbers.

        # This will underflow sometimes, but numpy is smart and will properly
        # return 0, as expected, in the cases where the probability is too small
        # to properly represent in floating point.
        p = self._total_error(z_to_use, z_stars)

        # So, really, the problem is with log. If e_hit is 0, we will get -inf.
        try:
            # Add the probability of this measurment in.
            p = np.log(p).sum()
        except:
            pdb.set_trace()
   
        return p


def test_beam_range_measurement():
    src_path_map = '../data/map/wean.dat'
    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map() 
    model = SensorModel(occupancy_map)

    fig = plt.figure()
    mng = plt.get_current_fig_manager();  # mng.resize(*mng.window.maxsize())
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])
   
    # x = 4650.0
    # y = 2200.0
    x = 1660.0
    y = 6100.0
    # x = 4833
    # y = 5111

    plt.plot([x / 10], [y / 10], marker='o', markersize=3, color="red")

    for deg in np.linspace(0, 360, num=500, endpoint=False):
        th = math.radians(deg)
        test_point = (np.array([x, y, th]))
        dist = model.beam_range_measurement(test_point)
        print "Distance of", dist, "for", deg, "degrees"

        x2 = x + dist * math.cos(th)
        y2 = y + dist * math.sin(th)
        plt.plot([x / 10, x2 / 10], [y / 10, y2 / 10], 'b-', linewidth=0.25)
    
    plt.show()


def test_error():
    src_path_map = '../data/map/wean.dat'
    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map() 
    model = SensorModel(occupancy_map)

    x = np.linspace(0, 8183, num=2500)
    y = model._total_error(x, np.ones(x.shape) * 4000)
    plt.plot(x, y)
    plt.axis([-100, 8283, 0, 1.25])
    plt.show()

    

 
if __name__=='__main__':
    # This conducts a bunch of tests on the Sensor Model
    test_beam_range_measurement()
    # test_error()
