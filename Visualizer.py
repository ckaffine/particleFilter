import math
import time

from matplotlib import pyplot as plt
from matplotlib import figure as fig
from matplotlib import animation as manimation

class Visualizer:

    def __init__(self, skips=10, header=""):
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Awesome Squad',
                        comment='Fite me!')
        self.writer = FFMpegWriter(fps=15, metadata=metadata)
        # Use timestamp for the name
        name = header + "_" + str(int(round(time.time()))) + ".mp4"
        self.writer.setup(plt.figure(1), name, dpi=100)

        self.frame_count = 0
        self.n = skips


    def visualize_map(self, occupancy_map):
        if self.frame_count % self.n != 0:
            return

        fig = plt.figure(1)
        plt.clf()

        mng = plt.get_current_fig_manager();  # mng.resize(*mng.window.maxsize())
        plt.ion()
        plt.imshow(occupancy_map, cmap='Greys')
        plt.axis([0, 800, 0, 800])
        plt.xlabel("X axis (10cm spacing)")
        plt.ylabel("Y axis (10cm spacing)")


    def visualize_timestep(self, X_bar, tstep):
        if self.frame_count % self.n != 0:
            return

        fig = plt.figure(1)
        x_locs = X_bar[:,0]/10.0
        y_locs = X_bar[:,1]/10.0
        scat_back = plt.scatter(x_locs, y_locs, c='r', marker='o', s=8)

        for particle in X_bar:
            x = particle[0] / 10
            y = particle[1] / 10
            th = particle[2]

            dist = 25
            x2 = x + dist * math.cos(th)
            y2 = y + dist * math.sin(th)

            plt.plot([x, x2], [y, y2], 'g-', linewidth=0.25)

        plt.pause(0.00001)


    def visualize_laser(self, odometry_laser, ranges):
        fig = plt.figure(2)
        plt.clf()
        x = odometry_laser[0]
        y = odometry_laser[1]
        th = odometry_laser[2]

        for i, dist in enumerate(ranges):
            x2 = x + dist * math.cos(math.radians(-90 + i) + th)
            y2 = y + dist * math.sin(math.radians(-90 + i) + th)
            plt.plot([x, x2], [y, y2], 'b-', linewidth=0.25)

        plt.plot([x], [y], marker='o', markersize=3, color="red")


    def visualize(self, occupancy_map, X_bar, tstep):
        self.visualize_map(occupancy_map)
        self.visualize_timestep(X_bar, tstep)

    
    def save(self):
        # Only actually save every nth frame
        self.frame_count += 1

        if self.frame_count % self.n != 0:
            return

        print "Saving a frame!"
        self.writer.grab_frame()


    def close(self):
        self.writer.finish()
