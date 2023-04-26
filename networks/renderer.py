# -*- coding: utf-8 -*-

# @Time:      2023/4/18 17:46
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      renderer.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
import torch
import torch.nn.functional as F
import numpy as np

from networks.neus.renderer import NeuSRenderer


# - Coding Part - #
class NeuSRendererPatch(NeuSRenderer):
    def __init__(self,
                 sdf_network,
                 deviation_network,
                 color_network,
                 n_samples=64,
                 n_importance=64,
                 up_sample_steps=4,
                 perturb=1.0):
        super(NeuSRendererPatch, self).__init__(sdf_network,
                                                deviation_network,
                                                color_network,
                                                n_samples,
                                                n_importance,
                                                up_sample_steps,
                                                perturb)

    def render_core(self,
                    cos_anneal_ratio=0.0,
                    patch_render=True,
                    **kwargs):

        # Parameters
        rays_o = kwargs['rays_o']
        rays_d = kwargs['rays_d']
        rays_d_patch = kwargs['rays_d_patch']  # [Nr, 3 * pch_len^2]
        z_vals = kwargs['z_vals']
        sample_dist = kwargs['sample_dist']

        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape).to(dists.device)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        sdf_nn_output = self.sdf_network(pts)
        sdf = sdf_nn_output[:, :1]

        gradients = self.sdf_network.gradient(pts).squeeze()

        if patch_render:
            # Computation:
            #       pts_nbr = pts_o + t * v_vec
            #       t = (pts_cen - pts_o) * n_vec / v_vec * n_vec
            # Dimension: [batch_num, n_samples, dim (3), pch_len^2].
            n_vec = gradients.reshape(batch_size, n_samples, 3, 1).detach()
            pts_cen = pts.reshape(batch_size, n_samples, 3, 1)
            pts_o = rays_o.reshape(batch_size, 1, 3, 1)
            v_vec = rays_d_patch.reshape(batch_size, 1, 3, -1)
            t1 = torch.sum((pts_cen - pts_o) * n_vec, dim=2, keepdim=True)
            t2 = torch.sum(v_vec * n_vec, dim=2, keepdim=True)
            t = t1 / t2  # [batch_size, n_samples, 1, -1]
            pts_nbr = pts_o + t * v_vec  # [batch_size, n_samples, 3, -1]
            pts_nbr_batch = pts_nbr.permute(0, 1, 3, 2).reshape(-1, 3)
            projected_color = self.color_network(pts_nbr_batch).reshape(batch_size, n_samples, -1)
        else:
            projected_color = self.color_network(pts).reshape(batch_size, n_samples, -1)

        # Compute alpha
        inv_s = self.deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)  # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)
        true_cos = (dirs * gradients).sum(-1, keepdim=True)
        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive
        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5
        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
        p = prev_cdf - next_cdf
        c = prev_cdf
        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]).to(alpha.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        # weights_sum = weights.sum(dim=-1, keepdim=True)

        rendered_color = (projected_color * weights[:, :, None]).sum(dim=1)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        # MOD: For warping loss
        pts_mat = pts.reshape(*weights.shape[:2], 3)
        pts_sum = (pts_mat * weights[:, :, None]).sum(dim=1)
        color_1pt = self.color_network(pts_sum)
        # color_1pt = reflect[:, :1] * self.color_network(pts_sum) + reflect[:, 1:]

        return {
            'rendered_color': rendered_color,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere,
            'color_1pt': color_1pt,
            'pts_sum': pts_sum,
        }

    def render(self,
               perturb_overwrite=-1,
               cos_anneal_ratio=0.0,
               patch_render=True,
               **kwargs):

        # Parameters
        rays_o = kwargs['rays_o']
        rays_d = kwargs['rays_d']
        rays_d_patch = kwargs['rays_d_patch']
        near = kwargs['near']
        far = kwargs['far']

        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples  # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples).to(rays_d.device)
        z_vals = near + (far - near) * z_vals[None, :]

        n_samples = self.n_samples

        perturb = self.perturb
        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5).to(z_vals.device)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)
                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2 ** i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_samples + self.n_importance

        # Render core
        ret_fine = self.render_core(rays_o=rays_o,
                                    rays_d=rays_d,
                                    rays_d_patch=rays_d_patch,
                                    z_vals=z_vals,
                                    sample_dist=sample_dist,
                                    cos_anneal_ratio=cos_anneal_ratio,
                                    patch_render=patch_render)

        rendered_color = ret_fine['rendered_color']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)

        return {
            'color_fine': rendered_color,
            's_val': s_val,
            'sdf': ret_fine['sdf'],
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere'],
            'color_1pt': ret_fine['color_1pt'],
            'pts_sum': ret_fine['pts_sum'],
        }
