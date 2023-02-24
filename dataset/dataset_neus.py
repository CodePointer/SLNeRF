# -*- coding: utf-8 -*-

# - Package Imports - #
import torch
import cv2
import numpy as np
import pointerlib as plb


# - Coding Part - #
class MultiPatDataset(torch.utils.data.Dataset):
    """Load image & pattern for one depth map"""
    def __init__(self, scene_folder, pat_idx_set, ref_img_set, sample_num, calib_para, device):
        self.scene_folder = scene_folder
        self.pat_folder = scene_folder / 'pat'
        self.img_folder = scene_folder / 'img'
        # self.depth_folder = scene_folder / 'gt'

        # Get pattern num
        self.pat_num = len(pat_idx_set)
        self.sample_num = sample_num
        self.intrinsics = plb.str2array(calib_para['img_intrin'], np.float32)
        self.intrinsics = plb.a2t(self.intrinsics).to(device)
        self.img_size = plb.str2tuple(calib_para['img_size'], item_type=int)
        self.img_size = (self.img_size[1], self.img_size[0])
        self.z_range = plb.str2tuple(calib_para['z_range'], float)

        # 读取img，pat，根据pat_idx_set。把它们stack到一起。
        img_list, pat_list = [], []
        for idx in pat_idx_set:
            img = plb.imload(self.img_folder / f'img_{idx}.png')
            img_list.append(img)
            pat = plb.imload(self.pat_folder / f'pat_{idx}.png')
            pat_list.append(pat)
        self.img_set = torch.cat(img_list, dim=0)  # [C, H, W]
        self.pat_set = torch.cat(pat_list, dim=0)  # [C, H, W]
        # self.depth = plb.imload(self.depth_folder / 'depth.png', scale=10.0)  # [1, H, W]

        # 读取reflect_set
        reflect_list = [plb.imload(self.img_folder / f'img_{idx}.png') for idx in ref_img_set]
        reflect_list[0] -= reflect_list[1]
        self.ref_set = torch.cat(reflect_list, dim=0)  # [2, H, W]

        self.device = device

        # Get coord candidate
        self.mask_occ = torch.ones([1, *self.img_size], dtype=torch.float32)
        if (scene_folder / 'gt' / 'mask_occ.png').exists():
            self.mask_occ = plb.imload(scene_folder / 'gt' / 'mask_occ.png').to(torch.float32)
        
        # To device
        self.img_set = self.img_set.to(device)    # [C, H, W]
        self.pat_set = self.pat_set.to(device)    # [C, H, W]
        self.ref_set = self.ref_set.to(device)    # [2, H, W]
        self.mask_occ = self.mask_occ.to(device)  # [1, H, W]

        # For efficiency: hei, wid range
        hei_set, wid_set = torch.where(self.mask_occ[0] > 0)
        self.wid_range = [wid_set.min().item(), wid_set.max().item()]
        self.hei_range = [hei_set.min().item(), hei_set.max().item()]

        # Other parameters from NeuS
        self.object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        self.object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        z_min, z_max = self.z_range
        fx, fy, dx, dy = self.intrinsics.cpu().numpy()
        hei_set, wid_set = torch.where(self.mask_occ[0] > 0)
        w_min, w_max = wid_set.min().item(), wid_set.max().item()
        h_min, h_max = hei_set.min().item(), hei_set.max().item()
        x_min = (w_min - dx) / fx * z_max
        x_max = (w_max - dx) / fx * z_max
        y_min = (h_min - dy) / fy * z_max
        y_max = (h_max - dy) / fy * z_max
        mid = np.array([0.5 * (x_max + x_min),
                        0.5 * (y_max + y_min),
                        0.5 * (z_max + z_min)], dtype=np.float32)  # pos
        scale = max((x_max - x_min), (y_max - y_min), (z_max - z_min)) / 2.0
        self.scale_mat = np.array([
            [scale, 0.0, 0.0, mid[0]],
            [0.0, scale, 0.0, mid[1]],
            [0.0, 0.0, scale, mid[2]],
            [0, 0, 0, 1.0]
        ], dtype=np.float32)
        intrinsic = np.array([
            [fx, 0.0, dx, 0.0], 
            [0.0, fy, dy, 0.0], 
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        self.intrinsics_inv = torch.inverse(plb.a2t(intrinsic)).to(device)
        self.pose_vec = torch.from_numpy(- mid / scale)
        self.rays_o = self.pose_vec[None, :].expand(self.sample_num, 3).to(device)  # batch_size, 3

    def __len__(self):
        return 1

    def gen_uniform_ray(self, resolution_level=1):
        l = resolution_level
        wid = self.wid_range[1] - self.wid_range[0]
        hei = self.hei_range[1] - self.hei_range[0]
        tx = torch.linspace(self.wid_range[0], self.wid_range[1], wid // l)
        ty = torch.linspace(self.hei_range[0], self.hei_range[1], hei // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        # pixels_x = pixels_x.reshape(-1)     # [W, H]
        # pixels_y = pixels_y.reshape(-1)     # [W, H]
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).to(self.device) # W, H, 3
        p = torch.matmul(self.intrinsics_inv[None, :, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_o = self.pose_vec[None, None, :].expand(rays_v.shape)

        reflect = self.ref_set[:, pixels_y.long(), pixels_x.long()]  # 2, W, H
        mask = self.mask_occ[:, pixels_y.long(), pixels_x.long()]  # 2, W, H
        # depth = self.depth[:, pixels_y.long(), pixels_x.long()]  # 2, W, H

        res = [x.to(self.device) for x in[
            rays_o.permute(2, 1, 0), rays_v.permute(2, 1, 0),
            reflect.permute(0, 2, 1), mask.permute(0, 2, 1),
            # depth.permute(0, 2, 1),
        ]]

        return res

    def gen_random_rays(self):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=self.wid_range[0], 
                                 high=self.wid_range[1],
                                 size=[self.sample_num])
        pixels_y = torch.randint(low=self.hei_range[0],
                                 high=self.hei_range[1],
                                 size=[self.sample_num])
        color = self.img_set[:, pixels_y, pixels_x].permute(1, 0)    # batch_size, 3
        mask = self.mask_occ[:, pixels_y, pixels_x].permute(1, 0)      # batch_size, 1
        ref = self.ref_set[:, pixels_y, pixels_x].permute(1, 0)     # MOD: batch_size, 2
        # depth = self.depth[:, pixels_y, pixels_x].permute(1, 0)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float().to(self.device)  # batch_size, 3
        p = torch.matmul(self.intrinsics_inv[:, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        # rays_o = self.pose_vec[None, :].expand(rays_v.shape)  # batch_size, 3
        
        # MOD：Return value
        res = [x.to(self.device) for x in [
            rays_v, color, mask, ref,  # depth
        ]]
        return res

    def get_scale_mat(self):
        return self.scale_mat

    def get_pat_set(self):
        return self.pat_set

    def get_cut_img(self, resolution_level):
        h_up, h_dn = self.hei_range
        w_lf, w_rt = self.wid_range
        img = self.img_set[:, h_up:h_dn, w_lf:w_rt]
        return torch.nn.functional.interpolate(img[None], scale_factor=1 / resolution_level)[0]

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def __getitem__(self, idx):
        # Generate random rays from one camera
        # img_hei, img_wid = self.img_set.shape[-2:]

        rays_v, color, mask, ref = self.gen_random_rays()

        ret = {
            # 'idx': torch.Tensor([idx]),
            'rays_o': self.rays_o,                          # [N, 3]
            'rays_v': rays_v,                               # [N, 3]
            'mask': mask,                                   # [N, 1]
            'color': color,                                 # [N, C]
            'reflect': ref,                                 # [N, 2]
            # 'pat': self.pat_set,                        # [C, Hp, Wp]
        }
        return ret
