from open3d import *

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

        self.pcd = PointCloud()
        self.pcd.points = Vector3dVector(self.points)

        self.vis = Visualizer()
        self.vis.create_window()
        self.vis.add_geometry(self.pcd)

    def depth2points(self, Z):
        self.points[:, 2] = np.ndarray.flatten(Z)


    def update(self, Z):
        # points = np.random.rand(60000,3)
        self.depth2points(Z)

        self.pcd.points = Vector3dVector(self.points)
        self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()