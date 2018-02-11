import numpy as np
import pdb

class Resampling:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """

    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """

    def multinomial_sampler(self, X_bar):

        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """

        """
        TODO : Add your code here
        """

        return X_bar_resampled

    def low_variance_sampler(self, X_bar):

        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """

        """
        TODO : Add your code here
        """

        # We have to normalize the weights here, since the sensor model doesn't
        # get all of the points at once. I suppose that could change ...

        # TODO: Verify that this is still correct when considering log
        # probabilities, since that is nonlinear.
        num_particles = len(X_bar) # This is the original number of particles, M
        keep = X_bar[:, 3] != -np.inf
        sliced = X_bar[keep, :] # Remove -inf weights
        total_weights = sliced[:, 3].sum()
        sliced[:, 3] /= total_weights # This is what we're sampling from

        if len(sliced) == 0:
            # If all the weights are infinity, then fix it.
            sliced = X_bar
            sliced[:, 3] = 1.0 / num_particles

        weights = sliced[:, 3]

        r = np.random.rand() * 1.0 / num_particles
        c = weights[0]
        i = 0
        for m in range(num_particles):
            U = r + (m - 1) * (1.0 / num_particles)
            while U > c:
                i += 1
                c += weights[i]
            X_bar[i, :] = sliced[i, :] # Assign back in, reuse X_bar

        return X_bar

if __name__ == "__main__":
    pass
