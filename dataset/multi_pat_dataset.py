# -*- coding: utf-8 -*-

# - Package Imports - #
import torch
import cv2
import numpy as np
import pointerlib as plb


# - Coding Part - #
class MultiPatDataset(torch.utils.data.Dataset):
    """Load image & pattern for one depth map"""
    def __init__(self, scene_folder, pat_idx_set, sample_num=None, calib_para=None, device=None):
        self.scene_folder = scene_folder
        self.pat_folder = scene_folder / 'pat'
        self.img_folder = scene_folder / 'img'
        self.depth_folder = scene_folder / 'depth'

        # Get pattern num
        self.pat_num = len(pat_idx_set)
        self.sample_num = sample_num
        self.intrinsics = plb.str2array(calib_para['img_intrin'], np.float32)
        self.intrinsics = plb.a2t(self.intrinsics).to(device)
        self.img_size = plb.str2tuple(calib_para['img_size'], item_type=int)

        # 读取img，pat，根据pat_idx_set。把它们stack到一起。
        img_list, pat_list = [], []
        for idx in pat_idx_set:
            img = plb.imload(self.img_folder / f'img_{idx}.png')
            img_list.append(img)
            pat = plb.imload(self.pat_folder / f'pat_{idx}.png')
            pat_list.append(pat)
        self.img_set = torch.cat(img_list, dim=0)  # [C, H, W]
        self.pat_set = torch.cat(pat_list, dim=0)  # [C, H, W]
        # self.depth = plb.imload(self.depth_folder / 'depth_0.png', scale=10.0)  # [1, H, W]

        self.device = device

        # Get coord
        self.rays_o = torch.zeros(size=[self.sample_num, 3], device=device)

        # Get coord candidate
        self.mask_occ = torch.ones([1, *self.img_size], dtype=torch.bool)
        if (scene_folder / 'mask' / 'mask_occ.png').exists():
            self.mask_occ = plb.imload(scene_folder / 'mask' / 'mask_occ.png').to(torch.bool)
        pixel_coord = np.stack(np.meshgrid(np.arange(self.img_size[1]), np.arange(self.img_size[0])), axis=0)
        pixel_coord = plb.a2t(pixel_coord, permute=False).reshape(2, -1)
        self.valid_coord = pixel_coord[:, self.mask_occ.reshape(-1)].to(torch.long)  # [2, N]

        # img_hei, img_wid = self.img_set.shape[-2:]
        # pixel_coords = np.stack(np.mgrid[:img_hei, :img_wid], axis=-1)[None, ...].astype(np.float32)
        # pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / img_hei
        # pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / img_wid
        # pixel_coords -= 0.5
        # self.coords = torch.from_numpy(pixel_coords).view(-1, 2)
        # self.coords = self.coords.to(device)

        self.img_set = self.img_set.to(device)
        self.pat_set = self.pat_set.to(device)
        # self.depth = self.depth.to(device)
        self.mask_occ = self.mask_occ.to(device)
        self.valid_coord = self.valid_coord.to(device)

    def __len__(self):
        return 1

    def get_bound(self):
        fx, fy, dx, dy = self.intrinsics
        # hei, wid = self.img_size
        wid, hei = self.img_size
        # z_min, z_max = 300.0, 900.0
        z_min, z_max = 500.0, 1500.0

        x_min = (0.0 - dx) / fx * z_max
        x_max = (wid - dx) / fx * z_max
        y_min = (0.0 - dy) / fy * z_max
        y_max = (hei - dy) / fy * z_max

        bound = torch.Tensor([
            [x_min, y_min, z_min],
            [x_max, y_max, z_max],
        ]).to(self.device)

        # center = (bound[0] + bound[1]) / 2.0
        # scale = (bound[1] - bound[0]) / 2.0

        return bound

    def get_uniform_ray(self, resolution_level=1, device=None):
        if device is None:
            device = self.device
        l = resolution_level
        img_hei, img_wid = self.img_set.shape[-2:]
        tx = torch.arange(0, img_wid, l, device=device) + l // 2
        ty = torch.arange(0, img_hei, l, device=device) + l // 2
        wid = tx.shape[0]
        hei = ty.shape[0]
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        pixels_x = pixels_x.reshape(-1)
        pixels_y = pixels_y.reshape(-1)

        color = self.img_set[:, pixels_y, pixels_x]  # [N, C]
        color = color.permute(1, 0)
        reflect = color[:, -2:]

        # Mash check
        mask_val = self.mask_occ[0, pixels_y, pixels_x]
        # pixels_x = pixels_x[mask_val]
        # pixels_y = pixels_y[mask_val]

        rays_v = self.pixel2ray(pixels_x, pixels_y)
        rays_o = torch.zeros_like(rays_v)
        return rays_o, rays_v, [hei, wid], mask_val, reflect

    def pixel2ray(self, pixels_x, pixels_y):
        fx, fy, dx, dy = self.intrinsics
        p = torch.stack([
            (pixels_x - dx) / fx,
            (pixels_y - dy) / fy,
            torch.ones_like(pixels_y)
        ], dim=1)  # [N, 3]
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # [N, 3]
        return rays_v

    def __getitem__(self, idx):
        # Generate random rays from one camera
        # img_hei, img_wid = self.img_set.shape[-2:]
        rand_args = dict(size=[self.sample_num], device=self.device)
        selected_idx = torch.randint(low=0, high=self.valid_coord.shape[1], **rand_args)
        selected_coord = self.valid_coord[:, selected_idx]
        pixels_x = selected_coord[0]
        pixels_y = selected_coord[1]

        color = self.img_set[:, pixels_y, pixels_x]  # [N, C]
        color = color.permute(1, 0)
        rays_v = self.pixel2ray(pixels_x, pixels_y)
        ret = {
            'idx': torch.Tensor([idx]),
            'rays_o': self.rays_o,  # [N, 3]
            'rays_v': rays_v,
            'color': color,
            'pat': self.pat_set,  # [C, Hp, Wp]
        }

        # ret = {
        #     'idx': torch.Tensor([idx]),
        #     'coord': self.coords,  # [Hi * Wi, 2]
        #     'img': self.img_set,   # [C, Hi, Wi]
        #     'pat': self.pat_set,   # [C, Hp, Wp]
        #     'depth': self.depth,   # [1, Hi, Wi]
        # }
        return ret
