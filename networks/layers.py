# -*- coding: utf-8 -*-

# - Package Imports - #
import torch
import numpy as np
import pointerlib as plb


# - Coding Part - #
class WarpFromDepth(torch.nn.Module):
    def __init__(self, calib_para, device=None):
        super().__init__()

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

        if device is not None:
            self.abc_mat = self.abc_mat.to(device)
            self.ext_tran = self.ext_tran.to(device)
            self.pat_intrin = self.pat_intrin.to(device)

        pass

    def forward(self, depth_mat, src_mat, mask_flag=False):
        """
            depth_mat: [N, 1, Hd, Wd]
            src_mat:  [N, C, Hs, Ws]
            mask_flag: Set to 0 when True.
        :return:
            warped_mat: [N, 1, Hd, Wd]
        """
        hei_s, wid_s = src_mat.shape[-2:]

        # Calculate uu, vv
        denominator = self.abc_mat[2] * depth_mat + self.ext_tran[2]
        xx = (self.abc_mat[0] * depth_mat + self.ext_tran[0]) / denominator
        yy = (self.abc_mat[1] * depth_mat + self.ext_tran[1]) / denominator
        fu, fv, du, dv = self.pat_intrin
        uu = fu * xx + du  # [N, 1, Hd, Wd]
        vv = fv * yy + dv  # [N, 1, Hd, Wd]

        # Warp by grid_sample
        uu_mat = uu.permute(0, 2, 3, 1)
        vv_mat = vv.permute(0, 2, 3, 1)
        xx_grid = 2.0 * uu_mat / (wid_s - 1) - 1.0
        yy_grid = 2.0 * vv_mat / (hei_s - 1) - 1.0
        xy_grid = torch.cat([xx_grid, yy_grid], dim=3)

        warped_mat = torch.nn.functional.grid_sample(src_mat, xy_grid, padding_mode='border', align_corners=False)
        if mask_flag:
            one_mat = torch.zeros_like(src_mat)
            mask = torch.nn.functional.grid_sample(one_mat, xy_grid, padding_mode='zeros', align_corners=False)
            mask[mask > 0.99] = 1.0
            mask[mask < 1.0] = 0.0
            warped_mat *= mask
        return warped_mat


class WarpFromXyz(torch.nn.Module):
    def __init__(self, calib_para, pat_mat, bound, device=None):
        """

        :param calib_para:
        :param pat_mat: [C, Hp, Wp]
        :param bound:
        :param device:
        """
        super().__init__()

        precision_type = np.float32
        self.img_size = plb.str2tuple(calib_para['img_size'], item_type=int)
        self.pat_size = plb.str2tuple(calib_para['pat_size'], item_type=int)
        pat_intrin = plb.str2array(calib_para['pat_intrin'], precision_type)
        ext_rot = plb.str2array(calib_para['ext_rot'], precision_type, [3, 3])
        ext_tran = plb.str2array(calib_para['ext_tran'], precision_type)

        self.ext_rot = plb.a2t(ext_rot).to(device).squeeze(0)
        self.ext_tran = plb.a2t(ext_tran).to(device)
        self.pat_intrin = plb.a2t(pat_intrin).to(device)

        self.bound = bound
        self.center_pt = (bound[1] + bound[0]) / 2.0
        self.center_pt = self.center_pt.reshape(1, -1)  # [1, 3]
        self.scale = (bound[1] - bound[0])
        self.scale = self.scale.reshape(1, -1)  # [1, 3]
        self.pat_mat = pat_mat.unsqueeze(0)  # [1, C, Hs, Ws]

        pass

    def forward(self, xyz_set, mask_flag=False):
        """
            xyz_set: [N, 3], range: [-1, 1]
            src_mat:  [N, C, Hs, Ws]
            mask_flag: Set to 0 when True.
        :return:
            warped_set: [N, 3]
        """
        xyz_cam = xyz_set * self.scale + self.center_pt  # [N, 3]
        xyz_pat = torch.matmul(self.ext_rot, xyz_cam[:, :, None]) + self.ext_tran[:, None]  # [N, 3, 1]
        xyz_pat = xyz_pat.squeeze(2)

        fu, fv, du, dv = self.pat_intrin
        pixels_u = fu * (xyz_pat[:, 0] / xyz_pat[:, 2]) + du
        pixels_v = fv * (xyz_pat[:, 1] / xyz_pat[:, 2]) + dv

        hei_s, wid_s = self.pat_mat.shape[-2:]

        uu = 2.0 * pixels_u / (wid_s - 1) - 1.0
        vv = 2.0 * pixels_v / (hei_s - 1) - 1.0
        uv_mat = torch.stack([uu, vv], dim=-1).reshape(1, -1, 1, 2)  # [1, N, 1, 2]

        sample_mat = torch.nn.functional.grid_sample(self.pat_mat, uv_mat, padding_mode='border', align_corners=False)
        if mask_flag:
            one_mat = torch.ones_like(self.pat_mat)
            mask = torch.nn.functional.grid_sample(one_mat, uv_mat, padding_mode='zeros', align_corners=False)
            mask[mask > 0.99] = 1.0
            mask[mask < 1.0] = 0.0
            sample_mat *= mask

        return sample_mat.squeeze(axis=3)[0].permute(1, 0)  # [N, C]

        # hei_s, wid_s = src_mat.shape[-2:]
        #
        # # Calculate uu, vv
        # denominator = self.abc_mat[2] * depth_mat + self.ext_tran[2]
        # xx = (self.abc_mat[0] * depth_mat + self.ext_tran[0]) / denominator
        # yy = (self.abc_mat[1] * depth_mat + self.ext_tran[1]) / denominator
        # fu, fv, du, dv = self.pat_intrin
        # uu = fu * xx + du  # [N, 1, Hd, Wd]
        # vv = fv * yy + dv  # [N, 1, Hd, Wd]
        #
        # # Warp by grid_sample
        # uu_mat = uu.permute(0, 2, 3, 1)
        # vv_mat = vv.permute(0, 2, 3, 1)
        # xx_grid = 2.0 * uu_mat / (wid_s - 1) - 1.0
        # yy_grid = 2.0 * vv_mat / (hei_s - 1) - 1.0
        # xy_grid = torch.cat([xx_grid, yy_grid], dim=3)
        #
        # warped_mat = torch.nn.functional.grid_sample(src_mat, xy_grid, padding_mode='border', align_corners=False)
        # if mask_flag:
        #     one_mat = torch.zeros_like(src_mat)
        #     mask = torch.nn.functional.grid_sample(one_mat, xy_grid, padding_mode='zeros', align_corners=False)
        #     mask[mask > 0.99] = 1.0
        #     mask[mask < 1.0] = 0.0
        #     warped_mat *= mask
        # return warped_mat


class PatternSampler(torch.nn.Module):
    def __init__(self, warp_layer):
        super().__init__()
        self.warp_layer = warp_layer

    def forward(self, points, reflects):
        point_color = self.warp_layer(points)
        # ref = reflects[:, :1]
        # bck = reflects[:, 1:]
        # return ref * point_color + bck
