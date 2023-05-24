# -*- coding: utf-8 -*-

# @Time:      2023/5/21 15:50
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      phase_visualization.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
from pathlib import Path
import torch
import numpy as np
from configparser import ConfigParser

import pointerlib as plb
from tools.fpp_toolkit.basic_coder import PMPCoder
from tools.fpp_toolkit import coord2depth


# - Coding Part - #
def create_pcd_from_phase(scene_folder):
    depth_gt = plb.imload(scene_folder / 'gt' / 'depth.png', scale=10.0)
    coord_x_gt = plb.imload(scene_folder / 'gt' / 'coord_x.png', scale=50.0)
    mask_gt = plb.imload(scene_folder / 'gt' / 'mask_occ.png')

    config = ConfigParser()
    config.read(str(scene_folder / 'config.ini'), encoding='utf-8')
    img_size = plb.str2tuple(config['RawCalib']['img_size'], int)
    coord_para = dict(
        cam_intrin=plb.str2tuple(config['RawCalib']['img_intrin'], float),
        pro_intrin=plb.str2tuple(config['RawCalib']['pat_intrin'], float),
        rot=plb.str2array(config['RawCalib']['ext_rot'], array_size=[3, 3]),
        tran=plb.str2array(config['RawCalib']['ext_tran'], array_size=[3]),
    )

    # Find all phase patterns
    period_set = set()
    possible_list = sorted(list((scene_folder / 'img').glob('img_pm*n3*.png')))
    for img_path in possible_list:
        img_name_stem = img_path.stem
        period_num = int(img_name_stem[6:img_name_stem.find('n3')])
        period_set.add(period_num)

    # Load
    img_set = {}
    for period_num in sorted(list(period_set)):
        img_set[period_num] = []
        for i in range(3):
            img_set[period_num].append(scene_folder / 'img' / f'img_pm{period_num}n3i{i}.png')

    # Decode
    phase_res = {}
    for period_num in img_set.keys():
        coder = PMPCoder(period_num)
        period_num_mat = (coord_x_gt // period_num).numpy()[0]
        phase_res[period_num] = coder.decode_imgs(img_set[period_num], period_num=period_num_mat)

    # Visualize
    depth_map = {}
    for period_num in phase_res:
        depth_map[period_num] = coord2depth(phase_res[period_num], **coord_para) * mask_gt.numpy()
    depth_map[0] = depth_gt.numpy()
    out_folder = scene_folder / 'out'
    out_folder.mkdir(exist_ok=True)
    for period_num in depth_map:
        v = plb.DepthVisualizer(img_size, coord_para['cam_intrin'])
        depth = depth_map[period_num]
        mask = (depth > 0.0).astype(np.float32)
        v.set_depth(depth, mask)
        v.update()
        v.get_pcd(out_folder / f'pcd_{period_num}.asc')

    print('Finished.')
    pass


def visualize_pcd(scene_folder):
    pass


def main():
    main_folder = Path('C:/SLDataSet/SLNeRF/5_Dataset0509')
    create_pcd_from_phase(main_folder / 'scene_00')
    pass


if __name__ == '__main__':
    main()
