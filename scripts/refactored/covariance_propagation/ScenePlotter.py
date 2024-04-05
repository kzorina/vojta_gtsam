import gtsam
import matplotlib.pyplot as plt
import numpy as np
import gtsam

class Plotter:

    def __init__(self, ax:plt.axis):
        self.ax:plt.axis = ax
        size = 1.5
        offset = (0, 0, 0.5)
        self.x_lim = (-size/2 + offset[0], size/2 + offset[0])
        self.y_lim = (-size/2 + offset[0], size/2 + offset[0])
        self.z_lim = (-size/2 + offset[0], size/2 + offset[0])
        self.initialize()

    def reset_default_lim(self):
        self.x_lim = self.ax.get_xlim()
        self.y_lim = self.ax.get_ylim()
        self.z_lim = self.ax.get_zlim()

    def initialize(self):
        self.ax.set_aspect('equal')
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_aspect('equal')

        self.ax.set_xlim(*self.x_lim)
        self.ax.set_ylim(*self.y_lim)
        self.ax.set_zlim(*self.z_lim)


    def plot_Q(self, Q, T:gtsam.Pose3, alpha=0.2, color='g'):
        assert Q.shape == (3, 3)
        # Q = (Q_ + Q_.T)/2
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        sphere = np.stack((x, y, z), axis=-1)[..., None]
        e, v = np.linalg.eigh(Q)
        s = v @ np.diag(np.sqrt(e)) @ v.T
        ellipsoid = (s @ sphere).squeeze(-1) + gtsam.Pose3(T).translation()
        self.ax.plot_surface(*ellipsoid.transpose(2, 0, 1), rstride=4, cstride=4, color=color, alpha=alpha)
        self.ax.plot_wireframe(*ellipsoid.transpose(2, 0, 1), rstride=4, linewidth=0.3, cstride=4, color='black', alpha=0.5)


    def plot_T(self, T_:gtsam.Pose3, size = 0.1, alpha=1):
        colors = ["red", "green", "blue"]
        T = T_.matrix()
        line_start = T[:3, 3]
        for i in range(3):
            line_end = line_start + T[:3, i]*size
            self.ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], [line_start[2], line_end[2]], c=colors[i], alpha=alpha)

    def plot_text(self, T:gtsam.Pose3, text:str):
        t = T.translation()
        self.ax.text(t[0], t[1], t[2], f"{text}", size=10, zorder=1,color='k')

    def plot_points(self, points):
        self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], c = 'b', marker='o')

    def set_camera_view(self, x=0, y=0,z=0):

        self.ax.view_init(x, y, z)
        # self.ax.update_layout(scene_camera=camera)

    def clear(self):
        self.ax.clear()
        self.initialize()

    def set_axes_equal(self):
        """
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc.

        Input
          ax: a matplotlib axis, e.g., as output from plt.gca().
        """

        x_limits = self.ax.get_xlim3d()
        y_limits = self.ax.get_ylim3d()
        z_limits = self.ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5 * max([x_range, y_range, z_range])

        self.ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        self.ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        self.ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])



if __name__ == "__main__":
    Q = np.diag((1, 1, 1, 10, 15, 17))
    T = gtsam.Pose3(np.array(((1, 0, 0, 1),
                              (0, 1, 0, 1),
                              (0, 0, 1, 0),
                              (0, 0, 0, 1))))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plotter = Plotter(ax)
    plotter.plot_Q(Q[3:6, 3:6], T)
    plotter.plot_T(T)
    # plotter.set_camera_view()
    plt.show()