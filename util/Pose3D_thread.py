import open3d as o3d
from open3d import *
import numpy.matlib
import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R
import copy
from util import ThreadICP
# from util import ProcessICP


class ClassVisPose3D:
    def __init__(self, n=100, m=200):
        self.n, self.m = n, m
        self.pixmm = 20  # pixel per mm
        # self.contact_z_threshold = 0.03
        self.contact_z_threshold = 0.02

        self.load_object()
        self.init_open3D()

        target = copy.deepcopy(self.obj)
        self.thread_icp = ThreadICP.ThreadICP(target, nb_worker=20, time_limit=0.005)
            
    def load_cube(self):  # object unit: mm
        # self.obj = o3d.io.read_point_cloud("objects/cube.ply")
        self.obj = o3d.io.read_point_cloud("objects/cube_20k.ply")
        # self.obj = o3d.io.read_point_cloud("objects/cylinder_20k.ply")
        self.obj.estimate_normals(fast_normal_computation=False)
        self.obj.paint_uniform_color([0.929, 0.651, 0])

        scale = np.eye(4)
        scale[3, 3] = self.m / self.pixmm * 2

        self.obj.transform(scale)

        r = R.from_euler("xyz", [37, 37, 0], degrees=True)
        trans = np.eye(4)
        trans[:3, :3] = r.as_matrix()
        trans[:3, 3] = [-0.0, -0.1, -1.2]
        trans[:3, 3] /= 2

        self.obj_id = "cube"

        self.obj_temp = copy.deepcopy(self.obj)
        self.obj_temp.transform(trans)
        self.trans = inv(trans)
        self.trans0 = self.trans.copy()

    def load_cylinder(self):  # object unit: mm
        # self.obj = o3d.io.read_point_cloud("objects/cube.ply")
        # self.obj = o3d.io.read_point_cloud("objects/cube_20k.ply")
        self.obj = o3d.io.read_point_cloud("objects/cylinder_20k.ply")
        self.obj.estimate_normals(fast_normal_computation=False)
        self.obj.paint_uniform_color([0.929, 0.651, 0])

        scale = np.eye(4)
        scale[3, 3] = self.m / self.pixmm / 1.5

        self.obj.transform(scale)

        r = R.from_euler("xyz", [0, 0, 0], degrees=True)
        trans = np.eye(4)
        trans[:3, :3] = r.as_matrix()
        trans[:3, 3] = [-0.0, -0.1, -1.2]
        trans[:3, 3] /= 2

        self.obj_id = "cylinder"

        self.obj_temp = copy.deepcopy(self.obj)
        self.obj_temp.transform(trans)
        self.trans = inv(trans)
        self.trans0 = self.trans.copy()

    def load_sphere(self):  # object unit: mm
        self.obj = o3d.io.read_point_cloud("objects/sphere_2k.ply")
        self.obj.estimate_normals(fast_normal_computation=False)
        self.obj.paint_uniform_color([0.929, 0.651, 0])

        scale = np.eye(4)
        # diameter: 12.66 mm
        scale[3, 3] = self.m / self.pixmm / 1.26

        self.obj.transform(scale)

        r = R.from_euler("xyz", [0, 0, 0], degrees=True)
        trans = np.eye(4)
        trans[:3, :3] = r.as_matrix()
        trans[:3, 3] = [-0.0, -0.1, -1.2]
        trans[:3, 3] /= 2

        self.obj_id = "sphere"

        self.obj_temp = copy.deepcopy(self.obj)
        self.obj_temp.transform(trans)
        self.trans = inv(trans)
        self.trans0 = self.trans.copy()

    def load_object(self):
        # self.load_cube()
        self.load_cylinder()
        # self.load_sphere()

    def init_open3D(self):
        x = np.arange(self.n) - self.n / 2
        y = np.arange(self.m) - self.m / 2
        self.X, self.Y = np.meshgrid(x, y)
        # Z = (X ** 2 + Y ** 2) / 10
        Z = np.sin(self.X)

        self.points = np.zeros([self.n * self.m, 3])
        self.points[:, 0] = np.ndarray.flatten(self.X) / self.m
        self.points[:, 1] = np.ndarray.flatten(self.Y) / self.m

        self.depth2points(Z)
        # exit(0)

        # points = np.random.rand(1,3)

        # self.pcd = PointCloud()

        self.pcd = o3d.geometry.PointCloud()

        self.pcd.points = o3d.utility.Vector3dVector(self.points)
        # self.pcd.colors = Vector3dVector(np.zeros([self.n, self.m, 3]))

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis.add_geometry(self.pcd)

        self.ctr = self.vis.get_view_control()
        self.ctr.change_field_of_view(-20)
        print("fov", self.ctr.get_field_of_view())
        self.ctr.convert_to_pinhole_camera_parameters()
        self.ctr.set_zoom(1)
        self.ctr.rotate(0, 450)  # mouse drag in x-axis, y-axis
        self.ctr.rotate(1050, 0)  # mouse drag in x-axis, y-axis
        self.vis.update_renderer()

    def depth2points(self, Z):
        self.points[:, 2] = np.ndarray.flatten(Z)

    def pose_reset(self):
        self.trans = self.trans0.copy()

    def icp(self, points):

        # contact map from depth
        mask = points[:, 2] > self.contact_z_threshold

        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(points[mask])
        source.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )

        # target = copy.deepcopy(self.obj)

        trans_init = self.trans
        trans_opt = trans_init.copy()

        trans_opt = self.thread_icp.estimate(copy.deepcopy(source), trans_init)

        # evaluation = o3d.registration.evaluate_registration(
        #     source, target, threshold, trans_init
        # )
        # print(evaluation)

        # self.trans = reg_p2l.transformation
        self.trans = trans_opt.copy()
        self.obj_temp = copy.deepcopy(self.obj)
        self.obj_temp.transform(inv(self.trans))

    def update(self, Z):
        Z = Z / self.m

        # points = np.random.rand(60000,3)
        self.depth2points(Z)

        points_sparse = self.points.reshape([self.n, self.m, 3])
        points_sparse = points_sparse[::1, ::1]
        points_sparse = points_sparse.reshape([-1, 3])

        print("Z.max()", Z.max())
        if Z.max() > self.contact_z_threshold * 1.5:
            self.icp(points_sparse)

        dx, dy = np.gradient(Z)
        dx, dy = dx * 300 / 2, dy * 300 / 2

        # I = np.array([-10,-1,-1])
        # I = I / np.sum(I**2)**0.5
        # # np_colors = (dy * I[0] + dx * I[1] - I[2]) / (dy ** 2 + dx ** 2 + 1) ** 0.5 * 3 + 0.5
        # np_colors = (dy * I[0] + dx * I[1] - I[2]) / (dy ** 2 + dx ** 2 + 1) ** 0.5
        # print("MIN MAX", np_colors.min(), np_colors.max())
        # np_colors = (np_colors - 0.5) * 20 + 0.5
        # np_colors[np_colors<0] = 0
        # np_colors[np_colors>1] = 1

        np_colors = dx + 0.5
        np_colors[np_colors < 0] = 0
        np_colors[np_colors > 1] = 1
        np_colors = np.ndarray.flatten(np_colors)
        colors = np.zeros([self.points.shape[0], 3])

        colors = np.zeros([self.points.shape[0], 3])

        for _ in range(3):
            colors[:, _] = np_colors
        # print("COLORS", colors)

        mask = self.points[:, 2] > self.contact_z_threshold
        obj_np_points = np.asarray(self.obj_temp.points)
        obj_np_colors = np.zeros([obj_np_points.shape[0], 3], dtype=np.float32)

        # print(obj_np_colors.shape)
        # print(self.obj_temp.has_normals())
        obj_normals = np.asarray(self.obj_temp.normals)
        print(obj_normals.shape)
        # light_vector = np.array([0, -1, -0.5])
        # light_vector = np.array([1, -2, -0.5])
        light_vector = np.array([0, -1, -0.5])
        light_vector = light_vector / (np.sum(light_vector ** 2)) ** 0.5
        diffusion = 0.6 + 0.5 * np.sum(obj_normals * light_vector, axis=-1)

        base_color = [1, 0.9, 0]
        if self.obj_id == "sphere":
            base_color = [0.95, 0.95, 0.95]
        if self.obj_id == "cylinder":
            base_color = [0.95, 0.95, 0.95]

        for c in range(3):
            obj_np_colors[:, c] = diffusion * base_color[c]
        # obj_np_colors[:, 0] = diffusion
        # obj_np_colors[:, 1] = diffusion * 0.9

        # self.pcd.points = o3d.utility.Vector3dVector(self.points)
        # self.pcd.colors = o3d.utility.Vector3dVector(colors)
        self.pcd.points = o3d.utility.Vector3dVector(
            np.vstack([self.points, self.obj_temp.points])
        )
        self.pcd.colors = o3d.utility.Vector3dVector(np.vstack([colors, obj_np_colors]))

        # mask = points_sparse[:, 2] > 0.03
        # self.pcd.points = o3d.utility.Vector3dVector(np.vstack([points_sparse[mask], self.obj_temp.points]))

        try:
            self.vis.update_geometry()
        except:
            self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
