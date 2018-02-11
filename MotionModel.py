import sys
import numpy as np
import math

class MotionModel:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """

    def __init__(self):

        """
        TODO : Initialize Motion Model parameters here
        """
        self.a1 = 1e-4
        self.a2 = 1e-4
        self.a3 = 1e-4
        self.a4 = 1e-4


    def error(self, var):
        # Return something error based on variance
        return np.random.normal(0, var ** 0.5, 1)


    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]   
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """

        """
        TODO : Add your code here
        """

        rot1 = math.atan2(u_t1[1] - u_t0[1], u_t1[0] - u_t0[0]) - u_t0[2]
        trans = math.sqrt((u_t1[1] - u_t0[1]) ** 2 + (u_t1[0] - u_t0[0]) ** 2)
        rot2 = u_t1[2] - u_t0[2] - rot1

        rot1_hat = rot1 + self.error(self.a1 * rot1 ** 2 + 
                                     self.a2 * trans ** 2)
        trans_hat = trans + self.error(self.a3 * trans ** 2 +
                                       self.a4 * rot1 ** 2 +
                                       self.a4 * rot2 ** 2)
        rot2_hat = rot2 + self.error(self.a1 * rot2 ** 2 +
                                     self.a2 * trans ** 2)

        # rot1_hat = rot1
        # trans_hat = trans
        # rot2_hat = rot2

        x = x_t0[0] + trans_hat * math.cos(x_t0[2] + rot1_hat)
        y = x_t0[1] + trans_hat * math.sin(x_t0[2] + rot1_hat)
        t = x_t0[2] + rot1_hat + rot2_hat
        t = (t + math.pi) % (2 * math.pi) - math.pi

        x_t1 = x_t0
        x_t1[0] = x
        x_t1[1] = y
        x_t1[2] = t
    
        return x_t1

if __name__=="__main__":
    pass
