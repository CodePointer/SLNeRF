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
from configparser import ConfigParser
import numpy as np
import cv2
from pathlib import Path
import shutil
from tqdm import tqdm

import pointerlib as plb


# - Coding Part - #
def visualize_point_cloud(config, pcd_name):
    calib_para = config['RawCalib']
    visualizer = plb.DepthVisualizer(
        img_size=plb.str2tuple(calib_para['img_size'], item_type=int),
        img_intrin=plb.str2tuple(calib_para['img_intrin'], item_type=float)
    )

    pos_center = [-50.0, 50.0, 50.0]
    rad = 40
    step = 32

    # visualizer.set_pcd(pcd_name)
    mask = plb.imload(pcd_name, scale=10.0, flag_tensor=False)
    mask = (mask > 0.0).astype(np.float32)
    visualizer.set_depth(pcd_name, mask)
    vis_folder = pcd_name.parent / 'vis_gif'
    vis_folder.mkdir(exist_ok=True, parents=True)
    for i in range(step):
        angle = np.pi * 2 / step * i
        dx = np.cos(angle) * rad
        dy = np.sin(angle) * rad
        visualizer.pos = [
            pos_center[0] + dx,
            pos_center[1] + dy,
            pos_center[2]
        ]
        visualizer.update()
        visualizer.get_view(vis_folder / f'{i}.png')

    # To gif
    plb.GifWriter(pcd_name.parent / f'{pcd_name.stem}.gif', fps=12).run_folder(vis_folder)
    shutil.rmtree(str(vis_folder))
    pass


def visualize_depth_map(folder):
    folder_out = folder.parent / 'vis' / 'depth_viz'
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
        cv2.imwrite(str(folder_out / f'depth_viz_e{epoch:05}.png'), depth_map[:640])

    for epoch in epoch_num:
        png_set = (folder / f'e_{epoch:05}').glob('(*).png')
        for png_file in png_set:
            png_out = folder.parent / 'vis' / png_file.stem
            png_out.mkdir(exist_ok=True, parents=True)
            shutil.copy(str(png_file), str(png_out / f'{png_file.stem}_e{epoch:05}{png_file.suffix}'))


def gif_write(folder):
    folders = plb.subfolders(folder.parent / 'vis')
    for x in folders:
        plb.GifWriter(x.parent / f'{x.name}.gif', fps=2).run_folder(x)


def main():
    # folder = Path(r'C:\SLDataSet\20221005realsyn-out\xyz2density-densitybest\output')
    # visualize_depth_map(folder)
    # gif_write(folder.parent / 'vis')

    # Draw GT
    pcd_names = [Path(x) for x in [
        # 'C:/SLDataSet/SLNeRF/1_Collected0218/scene_00/gt/pcd.xyz',
        # 'C:/SLDataSet/SLNeRF/1_Collected0218/scene_09/gt/pcd.xyz',
        # 'C:/SLDataSet/SLNeRF/1_Collected0218/scene_12/gt/pcd.xyz',
        # 'C:/SLDataSet/SLNeRF/1_Collected0218/scene_13/gt/pcd.xyz',
        # 'C:/SLDataSet/SLNeRF/1_Collected0218-out/scene_09/NPMP/output/e_50000/pcd.asc',
        # 'C:/SLDataSet/SLNeRF/1_Collected0218-out/scene_09/HPMP/output/e_50000/pcd.asc',
        # 'C:/SLDataSet/SLNeRF/1_Collected0218-out/scene_09/CGC7/output/e_50000/pcd.asc',
        # 'C:/SLDataSet/SLNeRF/1_Collected0218-out/scene_09/CGC6/output/e_50000/pcd.asc',
        # 'C:/SLDataSet/SLNeRF/1_Collected0218-out/scene_09/xyz2sdf-gc012345wrp/output/e_50000/pcd.asc',
        # 'C:/SLDataSet/SLNeRF/1_Collected0218-out/scene_13/NPMP/output/e_50000/pcd.asc',
        # 'C:/SLDataSet/SLNeRF/1_Collected0218-out/scene_13/HPMP/output/e_50000/pcd.asc',
        # 'C:/SLDataSet/SLNeRF/1_Collected0218-out/scene_13/CGC7/output/e_50000/pcd.asc',
        # 'C:/SLDataSet/SLNeRF/1_Collected0218-out/scene_13/CGC6/output/e_50000/pcd.asc',
        # 'C:/SLDataSet/SLNeRF/1_Collected0218-out/scene_13/xyz2sdf-gc012345wrp/output/e_50000/pcd.asc',
        # 'C:/SLDataSet/SLNeRF/1_Collected0218-out/scene_00/xyz2sdf-gc012345wrp/output/e_50000/pcd.asc',
        # 'C:/SLDataSet/SLNeRF/1_Collected0218-out/scene_00/xyz2sdf-gc0123wrp/output/e_50000/pcd.asc',
        # 'C:/SLDataSet/SLNeRF/1_Collected0218-out/scene_00/xyz2sdf-gc1234wrp/output/e_50000/pcd.asc',
        # 'C:/SLDataSet/SLNeRF/1_Collected0218-out/scene_00/xyz2sdf-gc2345wrp/output/e_50000/pcd.asc',
        # 'C:/SLDataSet/SLNeRF/1_Collected0218-out/scene_12/xyz2sdf-gc123456wrp/output/e_50000/pcd.asc',
        # 'C:/SLDataSet/SLNeRF/1_Collected0218-out/scene_12/xyz2sdf-gc123456wrp/output/e_25000/pcd.asc',
        'C:/SLDataSet/SLNeRF/1_Collected0218-out/scene_12/xyz2density-gc123456/output/e_50000/depth.png',
        # 'C:/SLDataSet/SLNeRF/1_Collected0218-out/scene_12/xyz2sdf-gc2345wrp/output/e_50000/pcd.asc',
        # 'C:/SLDataSet/SLNeRF/1_Collected0218-out/scene_12/xyz2sdf-gc2345wrp/output/e_25000/pcd.asc',
        'C:/SLDataSet/SLNeRF/1_Collected0218-out/scene_12/xyz2density-gc2345/output/e_50000/depth.png',
    ]]

    config = ConfigParser()
    config.read('C:/SLDataSet/SLNeRF/1_Collected0218/config.ini', encoding='utf-8')
    for pcd_name in tqdm(pcd_names):
        visualize_point_cloud(config, pcd_name)


if __name__ == '__main__':
    main()
