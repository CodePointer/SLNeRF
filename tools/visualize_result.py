# -*- coding: utf-8 -*-

# @Time:      2022/9/14 19:04
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      visualize_result.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
import open3d as o3d
import numpy as np
import cv2
from pathlib import Path

import pointerlib as plb


# - Coding Part - #
def visualize_point_cloud(folder):
    epoch_num = [int(x.name.split('_')[1]) for x in folder.glob('e_*')]
    epoch_num = sorted(epoch_num)
    print(epoch_num)

    epoch_folder = folder / f'e_{epoch_num[0]:05}'
    pcd_np = np.loadtxt(str(epoch_folder / 'depth_map.asc'), dtype=np.float32, delimiter=',', encoding='utf-8')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
    )
    o3d.visualization.draw_geometries([pcd])
    pass


def visualize_depth_map(folder):
    folder_out = folder.parent / 'depth_maps'
    folder_out.mkdir(exist_ok=True, parents=True)
    epoch_num = [int(x.name.split('_')[1]) for x in folder.glob('e_*')]
    epoch_num = sorted(epoch_num)
    for epoch in epoch_num:
        depth_name = folder / f'e_{epoch:05}' / 'depth_viz.png'
        depth_map = cv2.imread(str(depth_name), cv2.IMREAD_UNCHANGED)
        depth_map = cv2.putText(depth_map, f'epoch:{epoch}',
                                (50, 550),
                                cv2.FONT_HERSHEY_COMPLEX,
                                2, (255, 255, 255), thickness=4, lineType=cv2.LINE_8)
        plb.imviz(depth_map[:640], 'depth', 10)
        cv2.imwrite(str(folder_out / f'depth_e{epoch:05}.png'), depth_map[:640])


def main():
    folder = Path(r'C:\SLDataSet\20220907real-out\xyz2density-gray7bphase\output')
    visualize_depth_map(folder)


if __name__ == '__main__':
    main()
