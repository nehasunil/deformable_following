import open3d
from open3d import *
import numpy.matlib

class ClassVis3D:
    def __init__(self, n=100, m=200):
        self.n, self.m = n, m

        self.init_open3D()
        pass

    def init_open3D(self):
        x = np.arange(self.n)
        y = np.arange(self.m)
        self.X, self.Y = np.meshgrid(x,y)
        # Z = (X ** 2 + Y ** 2) / 10
        Z = np.sin(self.X)

        self.points = np.zeros([self.n * self.m, 3])
        self.points[:, 0] = np.ndarray.flatten(self.X) / self.m
        self.points[:, 1] = np.ndarray.flatten(self.Y) / self.m

        self.depth2points(Z)
        # exit(0)

        # points = np.random.rand(1,3)

        # self.pcd = PointCloud()

        self.pcd = open3d.geometry.PointCloud()

        self.pcd.points = open3d.utility.Vector3dVector(self.points)
        # self.pcd.colors = Vector3dVector(np.zeros([self.n, self.m, 3]))

        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis.add_geometry(self.pcd)

    def depth2points(self, Z):
        self.points[:, 2] = np.ndarray.flatten(Z)


    def update(self, Z):
        # points = np.random.rand(60000,3)
        self.depth2points(Z)

        dx, dy = np.gradient(Z)
        dx, dy = dx * 300, dy * 300

        # I = np.array([-10,-1,-1])
        # I = I / np.sum(I**2)**0.5
        # # np_colors = (dy * I[0] + dx * I[1] - I[2]) / (dy ** 2 + dx ** 2 + 1) ** 0.5 * 3 + 0.5
        # np_colors = (dy * I[0] + dx * I[1] - I[2]) / (dy ** 2 + dx ** 2 + 1) ** 0.5
        # print("MIN MAX", np_colors.min(), np_colors.max())
        # np_colors = (np_colors - 0.5) * 20 + 0.5
        # np_colors[np_colors<0] = 0
        # np_colors[np_colors>1] = 1

        np_colors = dx + 0.5
        np_colors[np_colors<0] = 0
        np_colors[np_colors>1] = 1
        np_colors = np.ndarray.flatten(np_colors)
        colors = np.zeros([self.points.shape[0], 3])

        colors = np.zeros([self.points.shape[0], 3])

        for _ in range(3): colors[:,_]  = np_colors
        # print("COLORS", colors)

        self.pcd.points = open3d.utility.Vector3dVector(self.points)
        self.pcd.colors = open3d.utility.Vector3dVector(colors)

        try:
            self.vis.update_geometry()
        except:
            self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
