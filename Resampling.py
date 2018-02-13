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
        self.prev_x_bar = None


    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """

        # take the average of how far the pose travelled, if it's less than half a meter then we shouldn't
        # resample
        resample_threshold_dist_cm = 100

        if self.prev_x_bar is None:
            self.prev_x_bar = X_bar

        pose_difference = X_bar - self.prev_x_bar;
        pose_difference = np.abs(pose_difference)
        avg_x_dist = np.mean(pose_difference[:, 0])
        avg_y_dist = np.mean(pose_difference[:, 1])

        avg_travel_dist = np.math.sqrt(avg_x_dist ** 2 + avg_y_dist ** 2)
        if avg_travel_dist < resample_threshold_dist_cm:
            return X_bar

        # Normalize the weights first to be compatible with the algorithm
        X_bar[:,3] /= X_bar[:,3].sum()

        X_bar_resampled = []

        M = len(X_bar)
        r = np.random.uniform(0, 1.0/M)
        c = X_bar[0][3]
        i = 0

        for m in range(M):
            U = r + (m)*(1.0/M)
            while U > c and i < M - 1:
                i = i + 1
                c = c + X_bar[i][3]
            X_bar_resampled.append(X_bar[i])

        # Re-normalize the weights again
        X_bar_resampled = np.array(X_bar_resampled)
        X_bar_resampled[:, 3] /= X_bar_resampled[:, 3].sum()

        self.prev_x_bar = X_bar_resampled

        assert len(X_bar_resampled) == M
        return X_bar_resampled


def test_identity_resample():
    data = np.random.rand(1000, 3)
    weights = np.ones((len(data), 1))
    X = np.hstack((data, weights))

    resampler = Resampling()
    X_bar = resampler.low_variance_sampler(X)

    assert((X_bar[:, :3] == data).all())


if __name__ == "__main__":
    test_identity_resample()
