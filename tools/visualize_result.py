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
def visualize_pcd_multiple(config, pcd_name_set, letter_set, output_file):
    calib_para = config['RawCalib']
    visualizer = plb.DepthVisualizer(
        img_size=plb.str2tuple(calib_para['img_size'], item_type=int),
        img_intrin=plb.str2tuple(calib_para['img_intrin'], item_type=float)
    )

    pos_center = [-50.0, 50.0, 50.0]
    rad = 40
    step_round = 32
    step_per_img = 4

    vis_folder = output_file.parent / 'vis_gif'
    vis_folder.mkdir(parents=True, exist_ok=True)
    step = 0
    for pcd_name, letter in zip(pcd_name_set, letter_set):
        visualizer.set_pcd(pcd_name)
        for _ in range(step_per_img):
            angle = np.pi * 2 / step_round * step
            step += 1
            dx = np.cos(angle) * rad
            dy = np.sin(angle) * rad
            visualizer.pos = [
                pos_center[0] + dx,
                pos_center[1] + dy,
                pos_center[2]
            ]
            visualizer.update()
            view = (visualizer.view * 255.0).astype(np.uint8)
            view_sub = cv2.resize(view, (320, 240))
            view_sub = cv2.putText(view_sub, str(letter), (10, 40), cv2.FONT_HERSHEY_COMPLEX,
                                   1, (0, 0, 0), 4, cv2.LINE_8)
            plb.imsave(vis_folder / f'{step}.png', view_sub, scale=1.0)

    # To gif
    plb.GifWriter(output_file, fps=20).run_folder(vis_folder)
    shutil.rmtree(str(vis_folder))


def visualize_point_cloud(config, pcd_name, letter=None, output_file=None):
    calib_para = config['RawCalib']
    visualizer = plb.DepthVisualizer(
        img_size=plb.str2tuple(calib_para['img_size'], item_type=int),
        img_intrin=plb.str2tuple(calib_para['img_intrin'], item_type=float)
    )

    pos_center = [-50.0, 50.0, 50.0]
    rad = 40
    step = 32

    visualizer.set_pcd(pcd_name)
    # mask = plb.imload(pcd_name, scale=10.0, flag_tensor=False)
    # mask = (mask > 0.0).astype(np.float32)
    # visualizer.set_depth(pcd_name, mask)
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
        view = (visualizer.view * 255.0).astype(np.uint8)
        view_sub = cv2.resize(view, (320, 240))
        view_sub = cv2.putText(view_sub, str(letter), (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 4, cv2.LINE_8)
        plb.imsave(vis_folder / f'{i}.png', view_sub, scale=1.0)
        # visualizer.get_view(vis_folder / f'{i}.png')

    # To gif
    if output_file is None:
        output_file = pcd_name.parent / f'{pcd_name.stem}.gif'
    plb.GifWriter(output_file, fps=20).run_folder(vis_folder)
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


def visualize_pattern_set(patterns, hei_num, wid_num, row_major=True, interval=20, canva_color=1.0):
    if len(patterns) == 0:
        return
    hei, wid = patterns[0].shape[-2:]
    for pat in patterns:
        h, w = pat.shape[-2:]
        if h != hei or w != wid:
            raise AssertionError('Input patterns are not in the same shape.')

    hei_all = hei * hei_num + interval * (hei_num + 1)
    wid_all = wid * wid_num + interval * (wid_num + 1)
    canva = np.ones([hei_all, wid_all], dtype=patterns[0].dtype) * canva_color

    iter_order = []
    if row_major:
        for h in range(hei_num):
            for w in range(wid_num):
                iter_order.append((h, w))
    else:
        for w in range(wid_num):
            for h in range(hei_num):
                iter_order.append((h, w))

    for i, (h, w) in enumerate(iter_order):
        if i < len(patterns):
            hei_start = h * (hei + interval) + interval
            wid_start = w * (wid + interval) + interval
            canva[hei_start:hei_start + hei, wid_start:wid_start + wid] = patterns[i]

    return canva


class EntrySet:

    @staticmethod
    def entry_pattern():
        # Draw pattern sets
        src_folder = Path(r'C:\SLDataSet\SLNeRF\8_Dataset0801\pat')
        dst_folder = Path(r'C:\SLDataSet\SLNeRF\8_Dataset0801')
        # pat_tags = [
        #     'pm48n3i0', 'pm48n3i1', 'pm48n3i2',
        #     'pm70n3i0', 'pm70n3i1', 'pm70n3i2',
        # ]
        # dst_name = 'npmp.png'
        # pat_tags = [
        #     'pm1280n3i0', 'pm1280n3i1', 'pm1280n3i2',
        #     'pm80n3i0', 'pm80n3i1', 'pm80n3i2',
        # ]
        # dst_name = 'hpmp.png'
        # pat_tags = [
        #     'gc0', 'gc1', 'gc2',
        #     'pm320n3i0', 'pm320n3i1', 'pm320n3i2',
        # ]
        # dst_name = 'gc6.png'
        # pat_tags = [
        #     'gc0', 'gc1', 'gc2', 'gc3',
        #     'pm160n3i0', 'pm160n3i1', 'pm160n3i2',
        # ]
        # dst_name = 'gc7.png'
        # pat_tags = [
        #     'gc0', 'gc1', 'gc2',
        #     'gc3', 'gc4', 'gc5',
        # ]
        # dst_name = 'gc6.png'
        # pat_tags = [
        #     'gc0', 'gc1', 'gc2',
        #     'gc3', 'gc4', 'gc5', 'gc6'
        # ]
        # dst_name = 'gc7.png'
        # pat_tags = [
        #     'arr20r0', 'arr20r1', 'arr10r0', 'arr10r1', 'arr5r0', 'arr5r1'
        # ]
        # dst_name = 'main6.png'
        # pat_tags = [
        #     'arr20r0', 'arr20r1', 'arr20r2',
        #     'arr10r0', 'arr10r1', 'arr10r2',
        #     'arr5r0', 'arr5r1', 'arr5r2'
        # ]
        # dst_name = 'online9.png'
        # pat_tags = [
        #     'gc0', 'gc1', 'gc2',
        #     'gc3', 'gc4', 'gc5',
        #     'gc6', 'gc7', 'gc8'
        # ]
        # dst_name = 'gc9.png'
        pat_tags = [
            'gc0inv', 'gc1inv', 'gc2inv',
            'gc3inv', 'gc4inv', 'gc5inv',
            'gc6inv', 'gc7inv', 'gc8inv'
        ]
        dst_name = 'gc9i.png'

        patterns = [plb.imload(x, flag_tensor=False) for x in [
            src_folder / f'pat_{tag}.png' for tag in pat_tags
        ]]
        res = visualize_pattern_set(patterns, 1, 9, row_major=False, interval=80, canva_color=0.8)
        plb.imsave(dst_folder / dst_name, res)

    @staticmethod
    def entry_point_cloud():
        data_folder = Path(r'C:\SLDataSet\SLNeRF\9_Dataset0923')
        exp_folder = Path(str(data_folder) + '-out')
        out_folder = data_folder / 'output'
        config = ConfigParser()
        config.read(str(data_folder / 'config.ini'), encoding='utf-8')
        pcd_list = []

        # GT
        # for i in range(1, 7):
        #     pcd_list.append(data_folder / f'scene_{i:02}' / 'gt' / 'pcd.xyz')

        # Classical
        # for i in range(2, 7):
        #     for exp_name in [
        #         'ClassicBFH-hpmp', 'ClassicBFN-npmp', 'ClassicGCC-cgc7',
        #         'ClassicGrayOnly-gcd7', 'xyz2sdf-main_20x2_10x2_5x2'
        #     ]:
        #         exp_set = exp_folder / f'scene_{i:02}' / exp_name
        #         for pcd_file in exp_set.rglob('pcd.asc'):
        #             pcd_list.append((pcd_file, out_folder / f'{exp_name}-{pcd_file.parent.name}.gif'))

        # # Loss
        # for i in range(1, 7):
        #     for exp_name in [
        #         'xyz2sdf-loss-rc', 'xyz2sdf-loss-rc-ek',
        #         'xyz2sdf-loss-sc', 'xyz2sdf-loss-sc-ek'
        #     ]:
        #         exp_set = exp_folder / f'scene_{i:02}' / exp_name / 'output' / 'iter_04000'
        #         for pcd_file in exp_set.rglob('pcd.asc'):
        #             pcd_list.append((pcd_file, out_folder / f'{exp_name}-{pcd_file.parent.name}.gif'))

        # GC
        # for i in range(1, 7):
        #     for gc_idx in range(3, 10):
        #         exp_set = exp_folder / f'scene_{i:02}' / f'ClassicGrayOnly-gcd{gc_idx}' / 'output' / 'iter_04000'
        #         pcd_list.append((exp_set / 'pcd.asc', out_folder / f's{i}_gc{gc_idx}.gif', gc_idx))
        #         exp_set = exp_folder / f'scene_{i:02}' / f'ClassicGrayOnly-gcd{gc_idx}inv' / 'output' / 'iter_04000'
        #         pcd_list.append((exp_set / 'pcd.asc', out_folder / f's{i}_gc{gc_idx}i.gif', gc_idx))
        # for i in range(1, 7):
        #     base_str = '0123456789'
        #     for gc_idx in range(3, 10):
        #         exp_set = exp_folder / f'scene_{i:02}' / f'xyz2sdf-gc{base_str[:gc_idx]}' / 'output' / 'iter_04000'
        #         pcd_list.append((exp_set / 'pcd.asc', out_folder / f'ours_s{i}_gc{gc_idx}.gif', gc_idx))
        #         exp_set = exp_folder / f'scene_{i:02}' / f'xyz2sdf-gc{base_str[:gc_idx]}inv' / 'output' / 'iter_04000'
        #         pcd_list.append((exp_set / 'pcd.asc', out_folder / f'ours_s{i}_gc{gc_idx}i.gif', gc_idx))

        # Self defined
        for exp_name in ['ClassicGrayOnly-gcd6', 'ClassicGrayOnly-gcd6-dyn']:
            exp_set = exp_folder / exp_name
            for pcd_file in exp_set.rglob('pcd.asc'):
                pcd_list.append((pcd_file, out_folder / f'{exp_name}-{pcd_file.parent.name}.gif'))

        for pcd, out_file, gc_idx in tqdm(pcd_list):
            visualize_point_cloud(config, pcd, gc_idx, out_file)

        pass

    @staticmethod
    def entry_point_cloud_multi():
        data_folder = Path(r'C:\SLDataSet\SLNeRF\8_Dataset0801')
        exp_folder = Path(str(data_folder) + '-out')
        out_folder = data_folder / 'output'
        config = ConfigParser()
        config.read(str(data_folder / 'config.ini'), encoding='utf-8')

        for i in range(1, 7):
            exp_set = exp_folder / f'scene_{i:02}' / 'xyz2sdf-main-online9-step' / 'output'
            tmp_mesh_list = [(x.parent.name.split('_')[-1][1:], x) for x in exp_set.rglob('pcd.asc')]
            tmp_pcd_list = sorted(tmp_mesh_list, key=lambda x: int(x[0]))
            pcd_list = [x[1] for x in tmp_pcd_list]
            letter_list = [x[0] for x in tmp_pcd_list]
            output_file = out_folder / f's{i}.gif'
            visualize_pcd_multiple(config, pcd_list, letter_list, output_file)


def main():
    EntrySet.entry_point_cloud()
    # EntrySet.entry_pattern()
    # EntrySet.entry_point_cloud_multi()
    pass


if __name__ == '__main__':
    main()
