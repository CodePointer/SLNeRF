# -*- coding: utf-8 -*-

# - Package Imports - #
import torch
import cv2
import numpy as np
import pointerlib as plb


# - Coding Part - #
class MultiPatDataset(torch.utils.data.Dataset):
    """Load image & pattern for one depth map"""
    def __init__(self, scene_folder, pat_idx_set, ref_img_set, sample_num, calib_para, device, rad=0):
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

        # 读取reflect_set
        reflect_list = [plb.imload(self.img_folder / f'img_{idx}.png') for idx in ref_img_set]
        reflect_list[0] -= reflect_list[1]
        self.ref_set = torch.cat(reflect_list, dim=0)  # [2, H, W]

        self.device = device

        # Get coord candidate
        # self.mask_occ = torch.ones([1, *self.img_size], dtype=torch.bool)
        # if (scene_folder / 'mask' / 'mask_occ.png').exists():
        #     self.mask_occ = plb.imload(scene_folder / 'mask' / 'mask_occ.png').to(torch.bool)
        # pixel_coord = np.stack(np.meshgrid(np.arange(self.img_size[1]), np.arange(self.img_size[0])), axis=0)
        # pixel_coord = plb.a2t(pixel_coord, permute=False).reshape(2, -1)
        # self.valid_coord = pixel_coord[:, self.mask_occ.reshape(-1)].to(torch.long)  # [2, N]

        self.pch_len = 2 * rad + 1
        pixel_coord = np.stack(np.meshgrid(np.arange(self.img_size[1]), np.arange(self.img_size[0])), axis=0)
        pixel_coord = plb.a2t(pixel_coord, permute=False).unsqueeze(0)  # [1, 2, H, W]
        pixel_unfold = torch.nn.functional.unfold(pixel_coord.float(), (self.pch_len, self.pch_len), padding=rad)  # [1, 2 * pch_len**2, H * W]
        self.mask_occ = torch.ones([1, *self.img_size], dtype=torch.float)
        if (scene_folder / 'mask' / 'mask_occ.png').exists():
            self.mask_occ = plb.imload(scene_folder / 'mask' / 'mask_occ.png')
        mask_unfold = torch.nn.functional.unfold(self.mask_occ.unsqueeze(0), (self.pch_len, self.pch_len), padding=rad)  # [1, 2 * pch_len**2, H * W]
        mask_bool = self.mask_occ.to(torch.bool).reshape(-1)  # [H * W]
        self.valid_patch = pixel_unfold[0, :, mask_bool].to(torch.long).reshape(2, self.pch_len, self.pch_len, -1).long()  # [2, pch_len, pch_len, L]
        self.valid_mask = mask_unfold[0, :, mask_bool].reshape(1, self.pch_len, self.pch_len, -1)  # [1, pch_len, pch_len, L]

        # Get coord
        self.rays_o = torch.zeros(size=[self.pch_len ** 2 * self.sample_num, 3], device=device)

        self.img_set = self.img_set.to(device)
        self.pat_set = self.pat_set.to(device)
        self.ref_set = self.ref_set.to(device)
        # self.depth = self.depth.to(device)
        self.mask_occ = self.mask_occ.to(device)
        # self.valid_coord = self.valid_coord.to(device)
        self.valid_patch = self.valid_patch.to(device)
        self.valid_mask = self.valid_mask.to(device)

    def __len__(self):
        return 1

    def get_pch_len(self):
        return self.pch_len

    def get_bound(self):
        fx, fy, dx, dy = self.intrinsics
        # wid, hei = self.img_size
        hei_set, wid_set = torch.where(self.mask_occ[0] > 0)
        w_min, w_max = wid_set.min(), wid_set.max()
        h_min, h_max = hei_set.min(), hei_set.max()
        # z_min, z_max = 300.0, 900.0
        z_min, z_max = 500.0, 1500.0

        x_min = (w_min - dx) / fx * z_max
        x_max = (w_max - dx) / fx * z_max
        y_min = (h_min - dy) / fy * z_max
        y_max = (h_max - dy) / fy * z_max

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

        # color = self.img_set[:, pixels_y, pixels_x].permute(1, 0)  # [N, C]
        reflect = self.ref_set[:, pixels_y, pixels_x].permute(1, 0)  # [pch_len**2 * N, 2]

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

        # # 上一个非patch的
        # selected_idx = torch.randint(low=0, high=self.valid_coord.shape[1], **rand_args)
        # selected_coord = self.valid_coord[:, selected_idx]

        selected_idx = torch.randint(low=0, high=self.valid_patch.shape[-1], **rand_args)
        selected_coord = self.valid_patch[:, :, :, selected_idx]  # [2, pch_len, pch_len, N]
        selected_mask = self.valid_mask[:, :, :, selected_idx]  # [1, pch_len, pch_len, N]

        pixels_x = selected_coord[1].reshape(-1)
        pixels_y = selected_coord[0].reshape(-1)
        color = self.img_set[:, pixels_y, pixels_x].permute(1, 0)  # [pch_len**2 * N, C]
        reflect = self.ref_set[:, pixels_y, pixels_x].permute(1, 0)  # [pch_len**2 * N, 2]

        fx, fy, dx, dy = self.intrinsics
        p = torch.stack([
            (pixels_x - dx) / fx,
            (pixels_y - dy) / fy,
            torch.ones_like(pixels_y)
        ], dim=1)  # [pch_len**2 * N, 3]
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # [pch_len**2 * N, 3]

        ret = {
            'idx': torch.Tensor([idx]),
            'rays_o': self.rays_o,                      # [N, 3]
            'rays_v': rays_v,                           # [pch_len**2 * N, 3]
            'mask': selected_mask.reshape(-1, 1),       # [pch_len**2 * N, 1]
            'color': color,                             # [pch_len**2 * N, C]
            'reflect': reflect,                         # [pch_len**2 * N, 2]
            'pat': self.pat_set,                        # [C, Hp, Wp]
        }

        # ret = {
        #     'idx': torch.Tensor([idx]),
        #     'coord': self.coords,  # [Hi * Wi, 2]
        #     'img': self.img_set,   # [C, Hi, Wi]
        #     'pat': self.pat_set,   # [C, Hp, Wp]
        #     'depth': self.depth,   # [1, Hi, Wi]
        # }
        return ret


class LCN(torch.nn.Module):
    """Local Contract Normalization"""
    def __init__(self, radius, epsilon):
        super().__init__()

        self.epsilon = epsilon
        self.radius = radius
        self.pix_num = (2 * self.radius + 1) ** 2

        self.avg_conv = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(radius),
            torch.nn.Conv2d(1, 1, kernel_size=2 * radius + 1, bias=False)
        )
        self.avg_conv[1].weight.requires_grad = False
        self.avg_conv[1].weight.fill_(1. / self.pix_num)

    def forward(self, data):
        self.avg_conv.to(data.device)
        avg = self.avg_conv(data)

        diff = data - avg
        std2 = self.avg_conv(diff ** 2)
        std = torch.sqrt(std2)

        img = (data - avg) / (std + self.epsilon)
        return img, std, avg


def mat2patch(input_mat, r):
    c, h, w = input_mat.shape
    side_len = 2 * r + 1
    res = torch.nn.functional.unfold(input_mat.reshape(c, 1, h, w).float(), (side_len, side_len),
                                     padding=r)
    res = res.reshape(-1, h, w)  # [C * pch_len^2, H, W]
    return res


class PatchLCNDataset(torch.utils.data.Dataset):
    """Load image & pattern for one depth map in patch"""
    def __init__(self, scene_folder, pat_idx_set, sample_num, calib_para, device, rad=7):
        self.scene_folder = scene_folder
        self.pat_folder = scene_folder / 'pat'
        self.img_folder = scene_folder / 'img'
        self.lcn_layer = LCN(rad, 1e-6)
        self.rad = rad

        # Get pattern num
        self.pat_num = len(pat_idx_set)
        self.sample_num = sample_num
        self.intrinsics = plb.str2array(calib_para['img_intrin'], np.float32)
        self.intrinsics = plb.a2t(self.intrinsics).to(device)
        self.img_size = plb.str2tuple(calib_para['img_size'], item_type=int)
        self.z_range = plb.str2tuple(calib_para['z_range'], float)

        # Load img & pat in ${pat_idx_set}. Stack them together.
        mat_set = {
            'img': [], 'pat': [],
            'mu': [], 'std': []
        }
        for idx in pat_idx_set:
            img = plb.imload(self.img_folder / f'img_{idx}.png')
            _, std, mu = self.lcn_layer(img[None])
            pat = plb.imload(self.pat_folder / f'pat_{idx}.png')
            mat_set['img'].append(img)
            mat_set['pat'].append(pat)
            mat_set['mu'].append(mu[0])
            mat_set['std'].append(std[0])
        self.mat_set = {x: torch.cat(mat_set[x]) for x in mat_set.keys()}

        # reflect_set
        reflect_list = [torch.ones_like(self.mat_set['img'][:1]), torch.zeros_like(self.mat_set['img'][:1])]
        self.ref_set = torch.cat(reflect_list, dim=0)  # [2, H, W]

        # Get mask
        self.mat_set['mask'] = torch.zeros([1, self.img_size[1], self.img_size[0]], dtype=torch.float)
        self.mat_set['mask'][:, rad:-rad, rad:-rad] = 1.0
        if (scene_folder / 'gt' / 'mask_occ.png').exists():
            self.mat_set['mask'] *= plb.imload(scene_folder / 'gt' / 'mask_occ.png')
        self.valid_coord = torch.stack(torch.where(self.mat_set['mask'][0] > 0), dim=0)

        self.patch_set = {key: mat2patch(self.mat_set[key], self.rad) for key in ['img', 'mask']}

        # To device
        self.device = device
        for x in self.mat_set:
            self.mat_set[x] = self.mat_set[x].to(device)
        self.ref_set = self.ref_set.to(device)

        # Other parameters from NeuS
        hei, wid = self.mat_set['mask'].shape[-2:]
        hei_set, wid_set = torch.where(self.mat_set['mask'][0] > 0)
        self.wid_range = [wid_set.min().item(), wid_set.max().item()]
        self.hei_range = [hei_set.min().item(), hei_set.max().item()]
        self.object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        self.object_bbox_max = np.array([1.01, 1.01, 1.01, 1.0])
        z_min, z_max = self.z_range
        fx, fy, dx, dy = self.intrinsics.cpu().numpy()
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

        # Pixel_xy with unfold. For patch-based computing.
        yy, xx = torch.meshgrid(
            torch.arange(0, hei),
            torch.arange(0, wid)
        )  # [H, W]
        p = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1).float().reshape(-1, 3).to(self.device)  # [H*W, 3]
        p = torch.matmul(self.intrinsics_inv[:, :3, :3], p[:, :, None]).squeeze()  # [H*W, 3]
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # [H*W, 3]
        self.mat_set['rays_v'] = rays_v.transpose(1, 0).reshape(3, hei, wid)  # [3, H, W]
        self.patch_set['rays_v'] = mat2patch(self.mat_set['rays_v'], self.rad)

    def __len__(self):
        return 1

    def gen_uniform_ray(self, resolution_level=1):
        l = resolution_level
        wid = self.wid_range[1] - self.wid_range[0]
        hei = self.hei_range[1] - self.hei_range[0]
        tx = torch.linspace(self.wid_range[0], self.wid_range[1], wid // l)
        ty = torch.linspace(self.hei_range[0], self.hei_range[1], hei // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)  # [W, H]

        rays_v = self.mat_set['rays_v'][:, pixels_y.long(), pixels_x.long()]  # 3, W, H
        rays_o = self.pose_vec[:, None, None].expand(rays_v.shape)  # 3, W, H

        reflect = self.ref_set[:, pixels_y.long(), pixels_x.long()]  # 2, W, H
        mask = self.mat_set['mask'][:, pixels_y.long(), pixels_x.long()]  # 2, W, H
        # depth = self.depth[:, pixels_y.long(), pixels_x.long()]  # 2, W, H

        res = [x.to(self.device) for x in [
            rays_o.permute(0, 2, 1), rays_v.permute(0, 2, 1),
            reflect.permute(0, 2, 1), mask.permute(0, 2, 1),
        ]]

        return res  # C, H, W

    def gen_random_rays(self):
        """
        Generate random rays at world space from one camera.
        """
        coord_idx = torch.randint(low=0, high=self.valid_coord.shape[1], size=[self.sample_num])
        pixels_y = self.valid_coord[0, coord_idx]
        pixels_x = self.valid_coord[1, coord_idx]
        # pixels_x = torch.randint(low=self.wid_range[0],
        #                          high=self.wid_range[1],
        #                          size=[self.sample_num])
        # pixels_y = torch.randint(low=self.hei_range[0],
        #                          high=self.hei_range[1],
        #                          size=[self.sample_num])

        color = self.mat_set['img'][:, pixels_y, pixels_x].permute(1, 0)  # [batch_size, C]
        mask = self.mat_set['mask'][:, pixels_y, pixels_x].permute(1, 0)  # [batch_size, 1]
        mu = self.mat_set['mu'][:, pixels_y, pixels_x].permute(1, 0)  # [batch_size, 1]
        std = self.mat_set['std'][:, pixels_y, pixels_x].permute(1, 0)  # [batch_size, 1]
        ref = self.ref_set[:, pixels_y, pixels_x].permute(1, 0)    # batch_size, 2
        rays_v = self.mat_set['rays_v'][:, pixels_y, pixels_x].permute(1, 0)  # [batch_size, 3]

        color_patch = self.patch_set['img'][:, pixels_y, pixels_x].permute(1, 0)  # [batch_size, C * pch^22]
        mask_patch = self.patch_set['mask'][:, pixels_y, pixels_x].permute(1, 0)  # [batch_size, pch^2]
        rays_v_patch = self.patch_set['rays_v'][:, pixels_y, pixels_x].permute(1, 0)  # [batch_size, 3 * pch^2]

        res = {
            'rays_v': rays_v.to(self.device),  # [N, 3]
            'rays_v_patch': rays_v_patch.to(self.device),  # [N, 3 * pch_num]
            'mask': mask.to(self.device),  # [N, 1]
            'mask_patch': mask_patch.to(self.device),  # [N, C * pch_num]
            'color': color.to(self.device),  # [N, C]
            'color_patch': color_patch.to(self.device),  # [N, C * pch_num]
            'reflect': ref.to(self.device),  # [N, 2]
            'mu': mu.to(self.device),  # [N, 1]
            'std': std.to(self.device),  # [N, 1]
        }
        return res

    def get_scale_mat(self):
        return self.scale_mat

    def get_pat_set(self):
        return self.mat_set['pat']

    def get_cut_img(self, resolution_level):
        h_up, h_dn = self.hei_range
        w_lf, w_rt = self.wid_range
        img = self.mat_set['img'][:, h_up:h_dn, w_lf:w_rt]
        return torch.nn.functional.interpolate(img[None], scale_factor=1 / resolution_level)[0]

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def get_pch_num(self):
        return (2 * self.rad + 1) ** 2

    def get_kernel(self, sigma):
        # Recommend: 10.0 -> 1.0 is enough.
        yy, xx = torch.meshgrid(
            torch.arange(-self.rad, self.rad + 1),
            torch.arange(-self.rad, self.rad + 1),
        )
        dist = torch.sqrt(xx.float() ** 2 + yy.float() ** 2)
        gaussian_val = torch.exp(- dist ** 2 / sigma ** 2)

        # kernel = gaussian_val / gaussian_val.sum()   This may cause numerical problems.
        kernel = gaussian_val
        return kernel.reshape(1, -1)  # [1, pch_len ** 2]

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
        ret = {
            'idx': torch.Tensor([idx]),
            'rays_o': self.pose_vec[None, :].expand(self.sample_num, 3).to(self.device),  # [N, 3]
            **self.gen_random_rays(),
        }
        return ret
