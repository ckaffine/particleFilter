import sys
import numpy as np
import math
import pdb

from matplotlib import pyplot as plt
from matplotlib import figure as fig

class MotionModel:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """

    def __init__(self, occupancy_map):

        self.a1 = 3e-5
        self.a2 = 3e-5
        self.a3 = 1e-1
        self.a4 = 1e-1

        # TODO: Do some convolution crap here
        self.map = occupancy_map
        self.max_y, self.max_x = self.map.shape
        self.max_y -= 1
        self.max_x -= 1
        self.min_y, self.min_x = 0, 0


    def error(self, var):
        # Return something error based on variance
        return np.random.normal(0, math.sqrt(var), 1)


    def _update(self, u_t0, u_t1, X_bar):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """

        # If the robot hasn't moved at all, we don't need to introduce error
        # into the positions of the particles.
        if (u_t0  == u_t1).all():
            return x_t0

        rot1 = math.atan2(u_t1[1] - u_t0[1], u_t1[0] - u_t0[0]) - u_t0[2]
        trans = math.sqrt((u_t1[1] - u_t0[1]) ** 2 + (u_t1[0] - u_t0[0]) ** 2)
        rot2 = u_t1[2] - u_t0[2] - rot1

        n = len(X_bar)

        rot1_hat = rot1 + np.random.normal(0, np.sqrt(self.a1 * rot1 ** 2 +
                                                      self.a2 * trans ** 2),
                                           (n, 1))

        trans_hat = trans + np.random.normal(0, np.sqrt(self.a3 * trans ** 2 +
                                                        self.a4 * rot1 ** 2 +
                                                        self.a4 * rot2 ** 2),
                                             (n, 1))
        rot2_hat = rot2 + np.random.normal(0, np.sqrt(self.a1 * rot2 ** 2 +
                                                      self.a2 * trans ** 2),
                                           (n, 1))

        x = X_bar[:, 0].reshape((n, 1))
        y = X_bar[:, 1].reshape((n, 1))
        t = X_bar[:, 2].reshape((n, 1))
        w = X_bar[:, 3].reshape((n, 1))

        x += trans_hat * np.cos(t + rot1_hat)
        x = np.minimum(x, 7990)
        x = np.maximum(x, 0)
        y += trans_hat * np.sin(t + rot1_hat)
        y = np.minimum(y, 7990)
        y = np.maximum(y, 0)
        t += rot1_hat + rot2_hat
        t = (t + math.pi) % (2 * math.pi) - math.pi

        return np.hstack((x, y, t, w))


    def get_val_from_coord(self, x_t):
        x_t = np.floor(x_t / 10).astype(int)
        x = x_t[0]
        y = x_t[1]
        return self.map[y][x]


    def update(self, u_t0, u_t1, x_t0):

        count = 0
        # Do this repeatedly until x_t1 is in the allowable space.
        while True:

            # If we loop too many times just give up.
            count += 1
            if count >= 10:
                return x_t0

            x_t1 = self._update(u_t0, u_t1, x_t0)
            return x_t1

            tx = int(math.floor(x_t1[0] / 10))
            ty = int(math.floor(x_t1[1] / 10))
            if (tx > self.max_x or tx < self.min_x or
                    ty > self.max_y or ty < self.min_y):
                continue

            if self.map[ty][tx] == 0:
                return x_t1


def visualize_robot_log(filename):
    # To tune the motion model, we'll walk through the robot log.

    fig = plt.figure()
    xs = []

    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            l = line.split()

            xs.append(map(float, l[1:4]))

    xs = np.array(xs)
    plt.plot(xs[:, 0], xs[:, 1], c='b')
    plt.plot(xs[0][0], xs[0][1], marker='o', markersize=3, color='red')
    plt.show()

if __name__=="__main__":
    for name in sys.argv[1:]:
        visualize_robot_log(name)

    if len(sys.argv) == 1:
        visualize_robot_log("../data/log/robotdata1.log")
