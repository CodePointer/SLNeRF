# -*- coding: utf-8 -*-

# @Time:      2022/6/21 17:45
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      main.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from pathlib import Path
from configparser import ConfigParser
from pointerlib import plb


# - Coding Part - #
class CoordCompute:
    def __init__(self, calib_para):
        precision_type = np.float32

        self.img_size = plb.str2tuple(calib_para['img_size'], item_type=int)
        self.pat_size = plb.str2tuple(calib_para['pat_size'], item_type=int)
        img_intrin = plb.str2array(calib_para['img_intrin'], precision_type)
        pat_intrin = plb.str2array(calib_para['pat_intrin'], precision_type)
        ext_rot = plb.str2array(calib_para['ext_rot'], precision_type, [3, 3])
        ext_tran = plb.str2array(calib_para['ext_tran'], precision_type)

        wid, hei = self.img_size
        fx, fy, dx, dy = img_intrin
        ww = np.arange(0, wid).reshape(1, -1).repeat(hei, axis=0)  # [hei, wid]
        hh = np.arange(0, hei).reshape(-1, 1).repeat(wid, axis=1)  # [hei, wid]
        vec_mat = np.stack([
            (ww - dx) / fx,
            (hh - dy) / fy,
            np.ones_like(ww),
        ], axis=2)
        vec_mat = vec_mat.reshape([hei, wid, 3, 1])

        self.abc_mat = np.matmul(ext_rot, vec_mat).reshape([hei, wid, 3])  # [hei, wid, 3]
        self.abc_mat = plb.a2t(self.abc_mat.astype(np.float32))  # [3, hei, wid]
        self.ext_tran = plb.a2t(ext_tran)
        self.pat_intrin = plb.a2t(pat_intrin)

    def __call__(self, depth):
        denominator = self.abc_mat[2] * depth + self.ext_tran[2]
        xx = (self.abc_mat[0] * depth + self.ext_tran[0]) / denominator
        yy = (self.abc_mat[1] * depth + self.ext_tran[1]) / denominator
        fu, fv, du, dv = self.pat_intrin
        uu = fu * xx + du
        vv = fv * yy + dv
        return uu, vv


class WarpLayer2D(torch.nn.Module):
    """Warping based on torch.grid_sample"""
    def __init__(self):
        super().__init__()
        pass

    def forward(self, uu_mat, vv_mat, src_mat, mask_flag=False):
        """src_mat: [... H, W]"""
        dst_hei, dst_wid = uu_mat.shape[-2:]
        src_hei, src_wid = src_mat.shape[-2:]
        src_mat = src_mat.reshape(-1, 1, src_hei, src_wid)
        batch = src_mat.shape[0]

        xx_grid = uu_mat.view(1, dst_hei, dst_wid, 1)
        yy_grid = vv_mat.view(1, dst_hei, dst_wid, 1)
        xx_grid = 2.0 * xx_grid / (src_wid - 1) - 1.0
        yy_grid = 2.0 * yy_grid / (src_hei - 1) - 1.0
        xy_grid = torch.cat([xx_grid, yy_grid], dim=3).expand(batch, dst_hei, dst_wid, 2)

        warped_mat = torch.nn.functional.grid_sample(src_mat, xy_grid, padding_mode='border', align_corners=False)
        if mask_flag:
            mask = torch.nn.functional.grid_sample(torch.ones_like(src_mat), xy_grid,
                                                   padding_mode='zeros', align_corners=False)
            mask[mask > 0.99] = 1.0
            mask[mask < 1.0] = 0.0
            warped_mat *= mask
        return warped_mat


def load_render(folder):  # TODO: 标定参数还是有很多小问题。先跳过。例如这里projector是先进行的t，再进行的R。
    # load parameter
    config = ConfigParser()
    config.read(str(folder / 'config.ini'), encoding='utf-8')

    # ext_tran = plb.str2array(config['Calibration']['ext_tran'], np.float32)
    # target_pt = np.array([0.0, 100.0, 800.0], dtype=np.float32)
    # look = target_pt - ext_tran
    # up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    # z_vec = look / np.linalg.norm(look)
    # right = np.cross(look, up)
    # x_vec = right / np.linalg.norm(right)
    # y_vec = np.cross(z_vec, x_vec)
    # rot_mat = np.stack([x_vec, y_vec, z_vec], axis=1)
    # rot_str = ','.join([str(x) for x in rot_mat.reshape(-1)])
    # config['Calibration']['ext_rot'] = rot_str

    depth2uv = CoordCompute(config['Calibration'])

    # load img, depth
    # scene_list = plb.subfolders(folder)

    scene_folder = folder
    depth = plb.imload(scene_folder / 'depth' / f'depth_0.png', scale=10.0).squeeze(0)
    uu, vv = depth2uv(depth)
    warp_layer = WarpLayer2D()
    pat_num = len(list((scene_folder / 'pat').glob('*.png')))
    for pat_idx in range(pat_num):
        pat = plb.imload(scene_folder / 'pat' / f'pat_{pat_idx}.png')
        img = warp_layer(uu, vv, pat, mask_flag=True)
        plb.imviz(img[0], 'img', 0)
        plb.imsave(scene_folder / 'img' / f'img_{pat_idx}.png', img[0])


# TODO: 先研究采样。
#  1. 看看现在的效果如何？ -> 目前来看还凑合。但原本的sample逻辑还是会有细节上的问题。
#  2. 考虑使用32个正常采样，剩下32个用来uniformly在最大值附近sample。
def calculate_occlusion(depth_mat):
    hei, wid = depth_mat.shape[-2:]
    mask_occ = torch.zeros_like(depth_mat)

    # TODO: 完成遮挡模式下的表示。可带可不带，可以先跳过这一步去下面一步。

    # depth_cam -> depth_pro，带去重

    # depth_pro -> mask_raw，用1来填充。

    # erode和dilate。参考pointerlib下面的data_generation.py

    # 展示mask，确定没有问题，保存。

    pass


# TODO: 完成mask之后进行渲染。有了结果之后放到原来的network里面进行训练，判断是否可行。

# TODO：采集数据，在真实场景上进行学习和训练。
def main():
    folder = Path(r'C:\SLDataSet\20220817s')

    depth = plb.imload(folder / 'depth' / 'depth_0.png', scale=10.0).squeeze(0)
    calculate_occlusion(depth)

    load_render(folder)
    pass


if __name__ == '__main__':
    main()
