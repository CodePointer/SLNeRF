# -*- coding: utf-8 -*-

# @Time:      2022/11/9 20:41
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      ply_to_depth.py
# @Software:  PyCharm
# @Description:
#   Convert ply file into depth map.

# - Package Imports - #
import open3d as o3d
from configparser import ConfigParser
from pathlib import Path
import numpy as np
import time

import pointerlib as plb


# - Coding Part - #
def ply2depth(ply_files, calib_para):
    wid, hei = plb.str2tuple(calib_para['img_size'], item_type=int)
    fx, fy, dx, dy = plb.str2tuple(calib_para['img_intrin'], item_type=float)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=wid, height=hei)
    inf_pinhole = o3d.camera.PinholeCameraIntrinsic(
        width=wid, height=hei,
        fx=fx, fy=fy, cx=dx, cy=dy
    )
    camera = o3d.camera.PinholeCameraParameters()
    camera.intrinsic = inf_pinhole
    camera.extrinsic = np.eye(4, dtype=np.float32)
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(camera, allow_arbitrary=True)

    depth_maps = []
    for ply_file in ply_files:
        mesh = o3d.io.read_triangle_mesh(str(ply_file))
        mesh.compute_vertex_normals()

        vis.add_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.01)

        depth = vis.capture_depth_float_buffer()
        depth = np.asarray(depth).copy()
        depth[np.isnan(depth)] = 0.0
        depth_maps.append(depth)

        vis.clear_geometries()

        # print(mesh)

    return depth_maps


def main():
    # Put all ply_files in to a list
    ply_path = Path('C:/Users/qiao/Desktop/CVPR2023_Sub/scene_03/NeuS')
    ply_files = list(ply_path.glob('*.ply'))

    # Load calib para
    config = ConfigParser()
    config.read(str(ply_path / 'config.ini'), encoding='utf-8')

    depth_maps = ply2depth(ply_files, config['Calibration'])
    # mask_map = None

    # Write all depth_maps
    for depth_map, ply_file in zip(depth_maps, ply_files):
        depth_file = ply_file.parent / f'{ply_file.stem}.png'
        plb.imsave(depth_file, depth_map, scale=10.0)

    pass


if __name__ == '__main__':
    main()
