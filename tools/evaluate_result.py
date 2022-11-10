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
    cell_set[0].value = f'{err10_num / total_num * 100.0:.2f}'
    cell_set[1].value = f'{err20_num / total_num * 100.0:.2f}'
    cell_set[2].value = f'{err50_num / total_num * 100.0:.2f}'
    cell_set[3].value = f'{avg:.3f}'


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


def process_scene(worksheet, params):
    data_path = Path(worksheet['B1'].value)
    res_path = Path(worksheet['B3'].value)
    depth_scale = 10.0 / worksheet['K1'].value

    # Get methods set
    all_value = [worksheet.cell(row=4, column=i + 1).value for i in range(worksheet.max_column)]
    methods = [x for x in all_value if x is not None]

    # Get experiments
    all_value = [worksheet.cell(row=i + 1, column=1).value for i in range(5, worksheet.max_row)]
    exps = [x for x in all_value if x is not None]

    # Load GT
    depth_gt = plb.imload(data_path / 'depth_map.png', scale=10.0)
    mask_gt = plb.imload(data_path / 'mask' / 'mask_occ.png')

    # Evaluate
    for met_i, method in enumerate(methods):
        for exp_i, exp in enumerate(exps):
            # Get cells
            # Start from B6
            row_i, col_i = exp_i + 6, met_i * 4 + 2
            cell_set = [worksheet.cell(row=row_i, column=col_i + x) for x in range(4)]
            flag_all_filled = False
            for cell in cell_set:
                flag_all_filled = flag_all_filled and cell.value is not None

            if params['clear'] and not flag_all_filled:
                for cell in cell_set:
                    cell.value = None

            # Check files
            depth_target_path = res_path / method / f'{exp}.png'
            if depth_target_path.exists():
                depth_target = plb.imload(depth_target_path, scale=10.0)

                if not params['recal'] and flag_all_filled:
                    continue

                evaluate(depth_gt * depth_scale, depth_target * depth_scale, mask_gt, cell_set)
            else:
                depth_target_path.parent.mkdir(parents=True, exist_ok=True)

    pass


def main():
    # Parameters
    params = {
        'recal': False,
        'clear': True,
        'workbook': 'C:/Users/qiao/Desktop/CVPR2023_Sub/Result.xlsx'
    }
    target_sheet_names = None

    # Load workbook and process sheets
    workbook = openpyxl.load_workbook(str(params['workbook']))
    if target_sheet_names is None:
        target_sheet_names = workbook.get_sheet_names()
    for scene_name in target_sheet_names:
        process_scene(workbook[scene_name], params)

    workbook.save(str(params['workbook']))

    # pat_name_list = [x[0].value for x in workbook['err1.0']['B3':'B14']]
    # square_size = float(workbook['err1.0']['R2'].value)
    # scale = 10.0 / square_size
    #
    # # Main experiments
    # for scene_num in range(5):
    #     scene_folder = main_folder / f'scene{scene_num:02}'
    #     if not scene_folder.exists():
    #         continue
    #     depth_gt = plb.imload(scene_folder / 'depth_map.png', scale=10.0)
    #     mask_occ = plb.imload(scene_folder / 'mask_occ.png')
    #     mask_occ[mask_occ > 0.99] = 1.0
    #     mask_occ[mask_occ < 1.0] = 0.0
    #
    #     for col_i, exp_name in enumerate(['GrayCode', 'Vanilla', 'Ours']):
    #         for row_i, pat_name in enumerate(pat_name_list):
    #             depth_exp_path = scene_folder / exp_name / f'{pat_name}.png'
    #             if not depth_exp_path.exists():
    #                 continue
    #             # Check if cell is empty
    #             row_idx, col_idx = row_i + 3, scene_num * 3 + col_i + 3
    #             cell_set = {x: workbook[x].cell(row_idx, col_idx) for x in sheet_names}
    #             flag_any_empty = flag_reset
    #             for _, cell in cell_set.items():
    #                 if cell.value is None:
    #                     flag_any_empty = True
    #
    #             if flag_any_empty:
    #                 depth_map = plb.imload(depth_exp_path, scale=10.0)
    #                 evaluate(depth_gt * scale, depth_map * scale, mask_occ, cell_set)
    #                 err_viz = draw_diff_viz(depth_gt * scale, depth_map * scale, mask_occ)
    #                 plb.imsave(scene_folder / exp_name / f'{pat_name}_diff.png', err_viz)
    #
    #     # Ablation Study
    #     if scene_num == 0:
    #         exp_names = [x.value for x in workbook['err1.0']['C19':'F19'][0]]
    #         for col_i, exp_name in enumerate(exp_names):
    #             for row_i, pat_name in enumerate(pat_name_list):
    #                 depth_exp_path = scene_folder / exp_name / f'{pat_name}.png'
    #                 if not depth_exp_path.exists():
    #                     continue
    #                 # Check if cell is empty
    #                 row_idx, col_idx = row_i + 20, scene_num + col_i + 3
    #                 cell_set = {x: workbook[x].cell(row_idx, col_idx) for x in sheet_names}
    #                 flag_any_empty = flag_reset
    #                 for _, cell in cell_set.items():
    #                     if cell.value is None:
    #                         flag_any_empty = True
    #
    #                 if flag_any_empty:
    #                     depth_map = plb.imload(depth_exp_path, scale=10.0)
    #                     evaluate(depth_gt * scale, depth_map * scale, mask_occ, cell_set)
    #                     err_viz = draw_diff_viz(depth_gt * scale, depth_map * scale, mask_occ)
    #                     plb.imsave(scene_folder / exp_name / f'{pat_name}_diff.png', err_viz)
    #
    # workbook.save(str(main_folder / 'CVPR2023Result.xlsx'))


if __name__ == '__main__':
    main()
