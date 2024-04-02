# -*- coding: utf-8 -*-

# @Time:      2024/2/29 22:21
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      mesh2depth.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
from pathlib import Path
from configparser import ConfigParser
import open3d as o3d
import numpy as np
import time
import cv2


# - Coding Part - #
class MyImgUtil:

    def __init__(self):
        pass

    @staticmethod
    def imload(path, scale=255.0, bias=0.0):
        """Load image with default type."""
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f'Failed to read file: {path}')
        img = (img.astype(np.float32) - bias) / scale
        return img

    @staticmethod
    def imsave(path, img, scale=255.0, bias=0.0, img_type=np.uint8, mkdir=False):
        """Save image."""
        img_copy = img.copy()
        img_copy = img_copy.astype(np.float32) * scale + bias
        # Check folder
        if mkdir:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img_copy.astype(img_type))


class MyO3DVisualizer:
    def __init__(self, img_size, img_intrin, pos, visible):
        self.img_size = img_size
        self.img_intrin = img_intrin
        self.pos = pos
        self.visible = visible

        self.vis = None
        self.ctr = None
        self.camera = None

    def __enter__(self):
        wid, hei = self.img_size
        fx, fy, dx, dy = self.img_intrin
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=wid, height=hei, visible=self.visible)
        inf_pinhole = o3d.camera.PinholeCameraIntrinsic(
            width=wid, height=hei,
            fx=fx, fy=fy, cx=dx, cy=dy
        )
        self.camera = o3d.camera.PinholeCameraParameters()
        self.camera.intrinsic = inf_pinhole
        self.camera.extrinsic = np.array([
            [1.0, 0.0, 0.0, self.pos[0]],
            [0.0, 1.0, 0.0, self.pos[1]],
            [0.0, 0.0, 1.0, self.pos[2]],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.ctr = self.vis.get_view_control()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.vis.clear_geometries()
        self.vis.destroy_window()
        if exc_type is not None:
            raise exc_type
        del self

    def set_pos(self, pos_4x4mat):
        self.camera.extrinsic = pos_4x4mat.astype(np.float64)

    def flush_pos(self):
        self.ctr.convert_from_pinhole_camera_parameters(self.camera, allow_arbitrary=True)

    def update(self):
        self.vis.poll_events()
        self.vis.update_renderer()
        time.sleep(0.01)

    def get_depth(self):
        depth = self.vis.capture_depth_float_buffer()
        depth = np.asarray(depth).copy()
        depth[np.isnan(depth)] = 0.0
        return depth

    def get_screen(self):
        img = self.vis.capture_screen_float_buffer()
        img = np.asarray(img).copy()
        return img


class DepthVisualizer:

    def __init__(self, img_size, img_intrin, pos=None, visible=False):
        self.img_intrin = img_intrin
        self.img_size = img_size

        self.depth = None
        self.mask = None
        self.pcd = None
        self.view = None

        self.pos = [0.0, 0.0, 0.0]
        if pos is not None:
            self.pos = pos
        self.visible = visible
        pass

    def set_mesh(self, ply):
        mesh = ply
        if isinstance(ply, Path) or isinstance(ply, str):
            mesh = o3d.io.read_triangle_mesh(str(ply))
        with MyO3DVisualizer(self.img_size, self.img_intrin,
                             [0.0, 0.0, 0.0], False) as v:
            mesh.compute_vertex_normals()
            v.vis.add_geometry(mesh)
            v.flush_pos()
            v.update()
            self.depth = v.get_depth()
            self.mask = (self.depth > 1.0).astype(np.float32)
        self.pcd = None
        self.view = None

    def set_depth(self, depth, mask, update=False):
        self.depth = depth
        if isinstance(depth, Path) or isinstance(depth, str):
            self.depth = MyImgUtil.imload(depth, 10.0)
        self.mask = mask
        if isinstance(mask, Path) or isinstance(mask, str):
            self.mask = MyImgUtil.imload(mask)

        self.pcd = None
        self.view = None

    def set_pcd(self, pcd):
        self.pcd = pcd
        if isinstance(pcd, Path) or isinstance(pcd, str):
            delimiter = ','
            if Path(pcd).suffix == '.asc':
                delimiter = ','
            elif Path(pcd).suffix == '.xyz':
                delimiter = None
            self.pcd = np.loadtxt(str(pcd), delimiter=delimiter, encoding='utf-8')

        self.depth = None
        self.mask = None
        self._pcd2depth()

    def update(self):
        self._depth2pcd()
        self._pcd2view()

    def get_depth(self, out_file=None):
        if out_file is None:
            return self.depth
        else:
            MyImgUtil.imsave(out_file, self.depth, 10.0, img_type=np.uint16)

    def get_pcd(self, out_file=None):
        if out_file is None:
            return self.pcd
        else:
            np.savetxt(str(out_file), self.pcd, fmt='%.2f',
                       delimiter=',', newline='\n', encoding='utf-8')

    def get_view(self, out_file=None):
        if out_file is None:
            return self.pcd
        else:
            MyImgUtil.imsave(out_file, self.view)

    def _depth2pcd(self):
        hei, wid = self.depth.shape[-2:]
        fx, fy, dx, dy = self.img_intrin
        hh = np.arange(0, hei).reshape(-1, 1).repeat(wid, axis=1)
        ww = np.arange(0, wid).reshape(1, -1).repeat(hei, axis=0)
        xyz_mat = np.stack([
            (ww - dx) / fx,
            (hh - dy) / fy,
            np.ones_like(ww)
        ], axis=0) * self.depth[None]
        xyz_set = xyz_mat.reshape(3, -1).transpose()
        self.pcd = xyz_set[self.mask.reshape(-1) > 0.0, :]

    def _pcd2depth(self):
        wid, hei = self.img_size
        fx, fy, dx, dy = self.img_intrin
        xx, yy, zz = np.split(self.pcd, 3, axis=1)
        ww = (xx / zz * fx + dx).astype(np.int32)
        hh = (yy / zz * fx + dy).astype(np.int32)
        ww[zz < 0.1] = 0
        hh[zz < 0.1] = 0

        idx = hh * wid + ww
        self.depth = np.zeros([hei, wid], dtype=np.float32).reshape(-1)
        np.put(self.depth, idx, zz)
        self.depth = self.depth.reshape([hei, wid])
        self.mask = (self.depth > 0.1).astype(np.float32)

    def _pcd2view(self):
        with MyO3DVisualizer(self.img_size, self.img_intrin, self.pos, False) as v:
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(self.pcd)
            pcd_o3d.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=30.0, max_nn=30)
            )
            pcd_o3d.colors = o3d.utility.Vector3dVector(np.ones_like(self.pcd) * 0.5)
            v.vis.add_geometry(pcd_o3d)
            v.flush_pos()
            v.update()
            self.view = v.get_screen()
        pass


def main():
    data_folder = Path('C:/Users/qiao/Desktop/EXP')

    # Load calibrated parameters
    config = ConfigParser()
    config.read(str(data_folder / 'config.ini'), encoding='utf-8')
    calib_para = config['RawCalib']

    def str2tuple(input_string, item_type=float):
        return tuple([item_type(x.strip()) for x in input_string.split(',')])

    # Create a visualizer
    visualizer = DepthVisualizer(
        img_size=str2tuple(calib_para['img_size'], item_type=int),
        img_intrin=str2tuple(calib_para['img_intrin'], item_type=float),
        pos=[-50.0, 50.0, 50.0]  # Only for visualized image.
    )

    # Input mesh
    visualizer.set_mesh(data_folder / 'mesh.ply')
    visualizer.update()

    # Get maps you need
    visualizer.get_depth(data_folder / 'depth.png')
    visualizer.get_view(data_folder / 'view.png')


if __name__ == '__main__':
    main()
