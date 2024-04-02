# -*- coding: utf-8 -*-

# @Time:      2023/4/16 20:08
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      test.py.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
from pathlib import Path
import re
import torch
from configparser import ConfigParser

import pointerlib as plb


# - Coding Part - #
def cut_image(scene_idx, iter_num, roi):
    data_path = 'C:/SLDataSet/SLNeRF/8_Dataset0801'
    out_path = f'{data_path}-out'

    def out_func(exp_tag, it=iter_num):
        return f'{out_path}/{exp_tag}/output/iter_{it:05}/{img_name}.png'

    # vis.png
    img_name = 'vis'
    vis_list = [
        ('gt', f'{data_path}/scene_{scene_idx:02}/gt/vis.png'),
        ('main', out_func('xyz2sdf-main_20x2_10x2_5x2')),
        ('loss-rc', out_func('xyz2sdf-loss-rc')),
        ('loss-sc', out_func('xyz2sdf-loss-sc')),
        ('loss-rcek', out_func('xyz2sdf-loss-rc-ek')),
        ('loss-scek', out_func('xyz2sdf-loss-sc-ek')),
        ('loss-rcsc', out_func('xyz2sdf-loss-rc-sc')),
        ('gc3', out_func('ClassicGrayOnly-gcd3')),
        ('gc4', out_func('ClassicGrayOnly-gcd4')),
        ('gc5', out_func('ClassicGrayOnly-gcd5')),
        ('gc6', out_func('ClassicGrayOnly-gcd6')),
        ('gc7', out_func('ClassicGrayOnly-gcd7')),
        ('gc8', out_func('ClassicGrayOnly-gcd8')),
        ('gc9', out_func('ClassicGrayOnly-gcd9')),
        ('gc3inv', out_func('ClassicGrayOnly-gcd3inv')),
        ('gc4inv', out_func('ClassicGrayOnly-gcd4inv')),
        ('gc5inv', out_func('ClassicGrayOnly-gcd5inv')),
        ('gc6inv', out_func('ClassicGrayOnly-gcd6inv')),
        ('gc7inv', out_func('ClassicGrayOnly-gcd7inv')),
        ('gc8inv', out_func('ClassicGrayOnly-gcd8inv')),
        ('gc9inv', out_func('ClassicGrayOnly-gcd9inv')),
        ('our-gc3', out_func('xyz2sdf-gc012')),
        ('our-gc4', out_func('xyz2sdf-gc012')),
        ('our-gc5', out_func('xyz2sdf-gc012')),
        ('our-gc6', out_func('xyz2sdf-gc012')),
        ('our-gc7', out_func('xyz2sdf-gc012')),
        ('our-gc8', out_func('xyz2sdf-gc012')),
        ('our-gc9', out_func('xyz2sdf-gc012')),
    ]


def cut_and_copy(scene_idx, vis_roi, img_roi):
    data_folder = Path(f'C:/SLDataSet/SLNeRF/8_Dataset0801/scene_{scene_idx:02}')
    out_folder = Path(f'C:/SLDataSet/SLNeRF/8_Dataset0801-out/scene_{scene_idx:02}')
    vis_folder = Path(f'C:/SLDataSet/SLNeRF/8_Dataset0801-out/out/scene_{scene_idx}_vis')
    vis_folder.mkdir(parents=True, exist_ok=True)
    err_folder = Path(f'C:/SLDataSet/SLNeRF/8_Dataset0801-out/out/scene_{scene_idx}_err')
    err_folder.mkdir(parents=True, exist_ok=True)

    vis_list = [
        ('gt', data_folder / 'gt' / 'vis.png')
    ]
    err_list = []

    # Example: C:\SLDataSet\SLNeRF\8_Dataset0801-out\scene_01\ClassicBFH-eval\output\iter_04000
    for exp_folder in out_folder.glob('*'):
        exp_tag = exp_folder.name
        for iter_folder in (exp_folder / 'output').glob('iter_*'):
            iter_num = int(iter_folder.name.split('_')[1])
            vis_list.append((f'{exp_tag}_{iter_num}', iter_folder / 'vis.png'))
            err_list.append((f'{exp_tag}_{iter_num}', iter_folder / 'err_vis.png'))

    # Cut
    for img_name, img_path in vis_list:
        h_src, w_src, h_end, w_end = vis_roi
        img = plb.imload(img_path, flag_tensor=False)
        img_cut = img[h_src:h_end, w_src:w_end]
        plb.imsave(vis_folder / f'{img_name}.png', img_cut)
    for img_name, img_path in err_list:
        h_src, w_src, h_end, w_end = img_roi
        img = plb.imload(img_path, flag_tensor=False)
        img_cut = img[h_src:h_end, w_src:w_end]
        plb.imsave(err_folder / f'{img_name}.png', img_cut)


def main():
    # Clip images
    # cut_and_copy(1, [122, 266, 722, 1066], [0, 300, 666, 1188])
    # cut_and_copy(2, [137, 238, 737, 1038], [0, 300, 666, 1188])
    # cut_and_copy(3, [130, 196, 730, 996], [0, 241, 666, 241 + 888])
    # cut_and_copy(4, [138, 244, 738, 1044], [0, 300, 666, 1188])
    # cut_and_copy(5, [135, 259, 735, 1059], [0, 320, 666, 1208])
    # cut_and_copy(6, [134, 259, 734, 1059], [0, 300, 666, 1188])

    # Scene 01
    # vis_list = [
    #     ('gt', 'C:/SLDataSet/SLNeRF/8_Dataset0801/scene_01/gt'),
    #     ('main_scene01', '')
    # ]

    data_folder = Path('C:/Users/qiao/Desktop/EXP')

    config = ConfigParser()
    config.read(str(data_folder / 'config.ini'), encoding='utf-8')
    calib_para = config['RawCalib']
    visualizer = plb.DepthVisualizer(
        img_size=plb.str2tuple(calib_para['img_size'], item_type=int),
        img_intrin=plb.str2tuple(calib_para['img_intrin'], item_type=float),
        pos=[-50.0, 50.0, 50.0],
    )

    visualizer.set_mesh(data_folder / 'mesh.ply')
    visualizer.get_depth(data_folder / 'depth.png')

    visualizer.update()
    visualizer.get_view(data_folder / 'view.png')

    pass


if __name__ == '__main__':
    main()
