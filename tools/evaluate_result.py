# -*- coding: utf-8 -*-

# @Time:      2022/11/1 15:40
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      evaluate_result.py
# @Software:  PyCharm
# @Description:
#   This file is used for depth map evaluation.

# - Package Imports - #
from configparser import ConfigParser
from pathlib import Path
import numpy as np
import cv2
import torch
import openpyxl

import pointerlib as plb


# - Coding Part - #
from dataset.multi_pat_dataset import MultiPatDataset
from networks.layers import WarpFromDepth


def evaluate(depth_gt, depth_map, mask, cell_set):
    diff = (depth_gt - depth_map)
    diff_vec = diff[mask > 0.0]
    total_num = diff_vec.shape[0]
    err10_num = (torch.abs(diff_vec) > 1.0).float().sum()
    err20_num = (torch.abs(diff_vec) > 2.0).float().sum()
    err50_num = (torch.abs(diff_vec) > 5.0).float().sum()
    avg = torch.abs(diff_vec).sum() / total_num
    cell_set['err1.0'].value = f'{err10_num / total_num * 100.0:.2f}'
    cell_set['err2.0'].value = f'{err20_num / total_num * 100.0:.2f}'
    cell_set['err5.0'].value = f'{err50_num / total_num * 100.0:.2f}'
    cell_set['avg'].value = f'{avg:.3f}'


def draw_depth_viz(depth):
    pass


def draw_diff_viz(depth_gt, depth_map, mask):
    diff = (depth_gt - depth_map)
    step_err = torch.ones_like(diff)
    step_err[torch.abs(diff) > 1.0] = 2.0
    step_err[torch.abs(diff) > 2.0] = 3.0
    step_err[torch.abs(diff) > 5.0] = 4.0
    step_vis = plb.VisualFactory.err_visual(step_err, mask, max_val=4.0, color_map=cv2.COLORMAP_WINTER)
    step_vis = cv2.cvtColor(plb.t2a(step_vis), cv2.COLOR_BGR2RGB)
    return step_vis


def main():
    flag_reset = False
    main_folder = Path('C:/SLDataSet/20221102real/CVPR2023')
    workbook = openpyxl.load_workbook(str(main_folder / 'CVPR2023Result.xlsx'))
    sheet_names = workbook.get_sheet_names()

    pat_name_list = [x[0].value for x in workbook['err1.0']['B3':'B14']]

    # Main experiments
    for scene_num in range(5):
        scene_folder = main_folder / f'scene{scene_num:02}'
        if not scene_folder.exists():
            continue
        depth_gt = plb.imload(scene_folder / 'depth_map.png', scale=10.0)
        mask_occ = plb.imload(scene_folder / 'mask_occ.png')
        mask_occ[mask_occ > 0.99] = 1.0
        mask_occ[mask_occ < 1.0] = 0.0

        for col_i, exp_name in enumerate(['GrayCode', 'Vanilla', 'Ours']):
            for row_i, pat_name in enumerate(pat_name_list):
                depth_exp_path = scene_folder / exp_name / f'{pat_name}.png'
                if not depth_exp_path.exists():
                    continue
                # Check if cell is empty
                row_idx, col_idx = row_i + 3, scene_num * 3 + col_i + 3
                cell_set = {x: workbook[x].cell(row_idx, col_idx) for x in sheet_names}
                flag_any_empty = flag_reset
                for _, cell in cell_set.items():
                    if cell.value is None:
                        flag_any_empty = True

                if flag_any_empty:
                    depth_map = plb.imload(depth_exp_path, scale=10.0)
                    evaluate(depth_gt, depth_map, mask_occ, cell_set)
                    err_viz = draw_diff_viz(depth_gt, depth_map, mask_occ)
                    plb.imsave(scene_folder / exp_name / f'{pat_name}_diff.png', err_viz)

        # Ablation Study
        if scene_num == 0:
            exp_names = [x.value for x in workbook['err1.0']['C19':'F19'][0]]
            for col_i, exp_name in enumerate(exp_names):
                for row_i, pat_name in enumerate(pat_name_list):
                    depth_exp_path = scene_folder / exp_name / f'{pat_name}.png'
                    if not depth_exp_path.exists():
                        continue
                    # Check if cell is empty
                    row_idx, col_idx = row_i + 20, scene_num + col_i + 3
                    cell_set = {x: workbook[x].cell(row_idx, col_idx) for x in sheet_names}
                    flag_any_empty = flag_reset
                    for _, cell in cell_set.items():
                        if cell.value is None:
                            flag_any_empty = True

                    if flag_any_empty:
                        depth_map = plb.imload(depth_exp_path, scale=10.0)
                        evaluate(depth_gt, depth_map, mask_occ, cell_set)
                        err_viz = draw_diff_viz(depth_gt, depth_map, mask_occ)
                        plb.imsave(scene_folder / exp_name / f'{pat_name}_diff.png', err_viz)

    workbook.save(str(main_folder / 'CVPR2023Result.xlsx'))


def test():
    depth_map = plb.imload('pat5m10.png', scale=10.0)
    mask_occ = plb.imload(r'C:\SLDataSet\20220907real\mask\mask_occ.png')

    config = ConfigParser()
    train_dir = Path('C:/SLDataSet/20220907real')
    config.read(str(train_dir / 'config.ini'), encoding='utf-8')

    pat_idx_set = [0, 1, 2, 3, 4]
    ref_img_set = [41, 40]
    my_device = torch.device('cpu')
    pat_dataset = MultiPatDataset(
        scene_folder=train_dir,
        pat_idx_set=pat_idx_set,
        ref_img_set=ref_img_set,
        sample_num=50,
        calib_para=config['Calibration'],
        device=my_device,
        rad=2
    )

    # warp_layer = WarpFromXyz(
    #     calib_para=config['Calibration'],
    #     pat_mat=self.pat_dataset.pat_set,
    #     bound=self.bound,
    #     device=self.device
    # )
    warp_layer = WarpFromDepth(
        calib_para=config['Calibration'],
        device=my_device
    )

    imgs = warp_layer(
        depth_mat=depth_map.unsqueeze(0),
        src_mat=pat_dataset.pat_set.unsqueeze(0),
    )

    plb.imviz_loop(imgs, 'imgs', 10)

    pass


if __name__ == '__main__':
    test()
