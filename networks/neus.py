# -*- coding: utf-8 -*-

# - Package Imports - #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import mcubes


# - Coding Part - #

# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    # This creation func was taken from NeuS implementation.
    # Ref: https://github.com/Totoro97/NeuS, models/embedder.py
    @staticmethod
    def create(multires, input_dims=3):
        embed_kwargs = {
            'include_input': True,
            'input_dims': input_dims,
            'max_freq_log2': multires - 1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }
        embedder_obj = Embedder(**embed_kwargs)
        def embed(x, eo=embedder_obj): return eo.embed(x)
        return embed, embedder_obj.out_dim

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = Embedder.create(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return x[:, :1] / self.scale, x[:, 1:]
        # return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)
        # return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            y, _ = self.sdf(x)
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
        return gradients.unsqueeze(1)

    def gradient_ray(self, z, ray):
        z.requires_grad_(True)
        with torch.enable_grad():
            x = ray[:, None, :] * z[..., :, None]
            y = self.sdf(x.reshape(-1, 3))
            y = torch.sigmoid(y)
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=z,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
        return gradients.unsqueeze(1)


class DensityNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(DensityNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = Embedder.create(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        val = torch.nn.functional.relu(x[:, :1] / self.scale)
        feature = x[:, 1:]
        return val, feature
        # return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)

    def sdf_hidden_appearance(self, x):
        return self.forward(x)


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class ReflectNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 warp_layer,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        self.squeeze_out = squeeze_out
        self.warp_layer = warp_layer
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = Embedder.create(multires_view, input_dims=d_in)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, points, feature_vectors, reflect=None):  # normals, view_dirs, feature_vectors):
        point_color = self.warp_layer(points)

        if self.embedview_fn is not None:
            points = self.embedview_fn(points)
            # view_dirs = self.embedview_fn(view_dirs)

        rendering_input = torch.cat([points, feature_vectors], dim=-1)

        # if self.mode == 'idr':
        #     rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        # elif self.mode == 'no_view_dir':
        #     rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        # elif self.mode == 'no_normal':
        #     rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            # x = torch.sigmoid(x)
            x = torch.tanh(x) / 2.0 + 0.5

        a = x[:, :1]
        b = x[:, 1:]

        # b = reflect[:, :1]
        # a = reflect[:, 1:] - b

        return a * point_color + b, x


# This code was taken from NeuS: https://github.com/Totoro97/NeuS
class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1], device=self.variance.device) * torch.exp(self.variance * 10.0)


# This code was taken from NeuS: https://github.com/Totoro97/NeuS
def extract_fields(normalize_func, vol_bound, resolution, query_func):
    N = 64
    bound_min, bound_max = vol_bound
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    # pts_set = []
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(bound_min.device)
                    # pts_set.append(pts.detach().cpu())
                    pts = normalize_func(pts)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    # pts_set = torch.cat(pts_set, dim=0)
    return u


# This code was taken from NeuS: https://github.com/Totoro97/NeuS
def extract_geometry(normalize_func, vol_bound, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(normalize_func, vol_bound, resolution, query_func)
    # min_z_idx = np.argmin(np.abs(u), axis=2).reshape(resolution, resolution, 1, 1).repeat(3, axis=3)
    # pts_volume = pts_set.reshape(resolution, resolution, resolution, -1)
    # pts_select = np.take_along_axis(pts_volume, min_z_idx, axis=2)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    bound_min, bound_max = vol_bound
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles  #, pts_set, pts_select.reshape(-1, 3)


# This code was taken from NeuS: https://github.com/Totoro97/NeuS
def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans

    # rad = 2
    # conv_win = 2 * rad + 1
    # kernel = torch.ones([1, 1, conv_win], dtype=torch.float32, device=bins.device) / conv_win
    # weights_in = weights.reshape(-1, 1, weights.shape[-1])
    # weights_out = torch.nn.functional.conv1d(weights_in, kernel, padding=0)
    # weights_out = torch.cat([weights_in[..., :rad], weights_out, weights_in[..., -rad:]], dim=-1)
    # weights = weights_out.reshape(weights.shape)

    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples, device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=bins.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def sample_pdf_avg(bins, weights, n_samples, det=False):
    """
    :param bins:    [1, K]
    :param weights:     [1, K]. The weight of every bin edges.
    :param n_samples:
    :param det:
    :return:
    """
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    weight_bin = (weights[..., :1] + weights[..., :-1]) * 0.5  # [1, K-1]
    pdf = weight_bin / torch.sum(weight_bin, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples, device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=bins.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def sample_pdf_uni(bins, weights, n_samples, alpha=1.0):
    """
    :param bins:    [1, K]
    :param weights:     [1, K]. The weight of every bin edges.
    :param n_samples:
    :param det:
    :return:
    """
    # max_idx for sample.
    sample_range = torch.arange(0, n_samples * alpha, alpha) - 0.5 * n_samples * alpha
    sample_range = sample_range.unsqueeze(0).to(bins.device)
    max_idx = torch.argmax(weights, dim=1)
    sample_val = bins[torch.arange(0, bins.shape[0]), max_idx].unsqueeze(1) + sample_range

    # Fill max, min
    min_set = bins[:, :1] + (sample_range + 0.5 * n_samples * alpha)
    max_set = bins[:, -1:] + (sample_range - 0.5 * n_samples * alpha)
    sample_min, _ = torch.min(sample_val, dim=1)
    sample_val[sample_min < bins[:, 0]] = min_set[sample_min < bins[:, 0]]
    sample_max, _ = torch.max(sample_val, dim=1)
    sample_val[sample_max > bins[:, -1]] = max_set[sample_max > bins[:, -1]]

    # Average sample for min == max
    max_min_thd = 0.01
    max_val, _ = torch.max(weights, dim=1)
    min_val, _ = torch.min(weights, dim=1)
    avg_mask = (max_val - min_val <= max_min_thd)
    mid_sample = torch.linspace(0.5 * (bins[0, 0] + bins[0, 1]), 0.5 * (bins[0, -2] + bins[0, -1]), n_samples)
    mid_sample = mid_sample.unsqueeze(0).to(bins.device)

    sample_val[avg_mask] = mid_sample

    return sample_val


# This code was modified based on NeuS: https://github.com/Totoro97/NeuS
class NeuSLRenderer:
    def __init__(self,
                 sdf_network,
                 deviation_network,
                 color_network,
                 n_samples,
                 n_importance,
                 # n_outside,
                 up_sample_steps,
                 perturb,
                 bound):
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        # self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb

        self.bound = bound

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1], device=rays_d.device), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1], device=rays_d.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    @staticmethod
    def density2weights(z_vals, density, z_steps=None):
        #
        # 这一段是原本NeRF的处理方式。
        #
        if z_steps is None:
            z_mids = (z_vals[:, :-1] + z_vals[:, 1:]) * 0.5
            z_bounds = torch.cat([
                z_vals[:, :1],
                z_mids,
                z_vals[:, -1:]
            ], dim=1)
            z_steps = z_bounds[:, 1:] - z_bounds[:, :-1]
        alpha = 1.0 - torch.exp(- density * z_steps)
        acc_trans = torch.cumprod(torch.cat([
            torch.ones_like(alpha[:, :1]),
            (1.0 - alpha + 1e-7),
        ], dim=1), dim=1)[:, :-1]
        weights = alpha * acc_trans

        #
        # 这里是Softmax，直接将density映射为weight。
        #
        # weights = torch.nn.functional.softmax(density, dim=1)
        return weights

    def up_sample_density(self, z_vals, density, n_importance):
        """
        Up sampling give a fixed inv_s
        """
        weights = self.density2weights(z_vals, density)
        if torch.any(torch.isnan(weights)):
            print(weights)
        z_samples = sample_pdf_avg(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def up_sample_uni(self, z_vals, density, n_importance, alpha):
        weights = self.density2weights(z_vals, density)
        z_samples = sample_pdf_uni(z_vals, weights, n_importance, alpha).detach()
        return z_samples

    def cat_z_vals(self, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            pts_n = self.pts_normalize(pts)
            new_sdf, _ = self.sdf_network.sdf(pts_n.reshape(-1, 3))
            new_sdf = new_sdf.reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    reflect,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0):
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape).to(rays_d.device)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        pts = self.pts_normalize(pts)
        dirs = dirs.reshape(-1, 3)

        sdf_nn_output = sdf_network(pts)
        sdf = sdf_nn_output[0]
        # sdf = sdf_nn_output[:, :1]
        # feature_vector = sdf_nn_output[:, 1:]

        gradients = sdf_network.gradient(pts).squeeze()
        # sampled_color = color_network(pts).reshape(batch_size, n_samples, -1)
        projected_color = color_network(pts).reshape(batch_size, n_samples, -1)
        sampled_color = reflect[:, None, :1] * projected_color + reflect[:, None, 1:]

        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
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

        # Render with background
        # if background_alpha is not None:
        #     alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
        #     alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
        #     sampled_color = sampled_color * inside_sphere[:, :, None] +\
        #                     background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
        #     sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1], device=rays_d.device),
                                                   1. - alpha + 1e-7], -1), -1)[:, :-1]
        # weights_sum = weights.sum(dim=-1, keepdim=True)

        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        # if background_rgb is not None:    # Fixed background, usually black
        #     color = color + background_rgb * (1.0 - weights_sum)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        return {
            'color': color,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere
        }

    def pts_normalize(self, pts):
        bound_min, bound_max = self.bound
        scale = (bound_max - bound_min) * 0.5
        center_pt = (bound_max + bound_min) / 2.0
        pts_normalized = (pts - center_pt) / scale
        return pts_normalized

    def render(self, rays_o, rays_d, bound, reflect, background_rgb=None, cos_anneal_ratio=0.0):
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples, device=rays_d.device)
        near, far = bound[0, 2], bound[1, 2]
        z_vals = near + (far - near) * z_vals[None, :]

        # z_vals_outside = None
        # if self.n_outside > 0:
        #     z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1], device=rays_d.device) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            # if self.n_outside > 0:
            #     mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
            #     upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
            #     lower = torch.cat([z_vals_outside[..., :1], mids], -1)
            #     t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
            #     z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        # if self.n_outside > 0:
        #     z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                pts_n = self.pts_normalize(pts)
                sdf, _ = self.sdf_network.sdf(pts_n.reshape(-1, 3))
                sdf = sdf.reshape(batch_size, self.n_samples)

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_samples + self.n_importance

        # Background model
        # if self.n_outside > 0:
        #     z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
        #     z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
        #     ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)
        #
        #     background_sampled_color = ret_outside['sampled_color']
        #     background_alpha = ret_outside['alpha']

        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    reflect,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio)

        color_fine = ret_fine['color']
        weights = ret_fine['weights']
        mid_z_vals = ret_fine['mid_z_vals']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)

        return {
            'color_fine': color_fine,
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'mid_z_vals': mid_z_vals,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere']
        }

    def render_density(self, rays_d, reflect, alpha=None):
        batch_size = len(rays_d)
        z_vals = torch.linspace(0.0, 1.0, self.n_samples, device=rays_d.device)
        near, far = self.bound[0, 2], self.bound[1, 2]
        z_vals = near + (far - near) * z_vals[None, :]

        perturb = self.perturb
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1], device=rays_d.device) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_d[:, None, :] * z_vals[..., :, None]
                pts_n = self.pts_normalize(pts)
                sdf, _ = self.sdf_network.sdf(pts_n.reshape(-1, 3))
                sdf = sdf.reshape(batch_size, self.n_samples)

                if alpha is not None:
                    new_z_vals = self.up_sample_uni(z_vals, sdf, self.n_importance, alpha)
                    z_vals, sdf = self.cat_z_vals(rays_d, z_vals, new_z_vals, sdf, last=True)
                else:  # Without sampling strategy
                    for i in range(self.up_sample_steps):
                        new_z_vals = self.up_sample_density(z_vals,
                                                            sdf,
                                                            self.n_importance // self.up_sample_steps)
                        z_vals, sdf = self.cat_z_vals(rays_d,
                                                      z_vals,
                                                      new_z_vals,
                                                      sdf,
                                                      last=(i + 1 == self.up_sample_steps))

        # Render core
        batch_size, n_samples = z_vals.shape  # [N, C]

        # Section midpoints
        pts = rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        pts_n = self.pts_normalize(pts.reshape(-1, 3))
        density, features = self.sdf_network(pts_n)
        density = density.reshape(z_vals.shape)

        #
        # Compute colors
        #
        projected_color = self.color_network(pts_n).reshape(batch_size, n_samples, -1)
        sampled_color = reflect[:, None, :1] * projected_color + reflect[:, None, 1:]
        weights = self.density2weights(z_vals=z_vals, density=density)
        color = (sampled_color * weights[:, :, None]).sum(dim=1)

        # Compute depth
        pts_sum = (pts * weights[:, :, None]).sum(dim=1)
        color_1pt = reflect[:, :1] * self.color_network(self.pts_normalize(pts_sum)) + reflect[:, 1:]
        depth_val = pts_sum[:, -1:]

        return {
            'pts': pts,
            'pt_color': sampled_color,
            'color': color,
            'color_1pt': color_1pt,
            'depth': depth_val,
            'density': density,
            'z_vals': z_vals,
            'weights': weights,
        }

    def extract_geometry(self, vol_bound, resolution, threshold=0.0):
        return extract_geometry(normalize_func=self.pts_normalize,
                                vol_bound=vol_bound,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts)[0])

    def extract_proj_geometry(self, rays_d, img_size, bound, z_resolution, threshold=0.0):
        return extract_proj_geometry(normalize_func=self.pts_normalize,
                                     rays_d=rays_d,
                                     img_size=img_size,
                                     bound=bound,
                                     z_resolution=z_resolution,
                                     threshold=threshold,
                                     query_func=lambda pts: -self.sdf_network.sdf(pts))


def extract_proj_geometry(normalize_func, rays_d, img_size, bound, z_resolution, threshold, query_func):
    # print('threshold: {}'.format(threshold))

    # 1. Extract projected fields
    sample_n = 64

    hei, wid = img_size
    u_rays = torch.zeros([wid * hei, z_resolution])
    z_vals = torch.linspace(0.0, 1.0, z_resolution, device=rays_d.device)
    near, far = bound[0, 2], bound[1, 2]
    near = near + 20.0
    far = far - 20.0
    z_vals = near + (far - near) * z_vals[None, :]
    rays_d_set = rays_d.split(sample_n)

    pts_set = []
    with torch.no_grad():
        for ui, rays_d in enumerate(rays_d_set):
            pts = rays_d[:, None, :] * z_vals[..., :, None]
            pts_set.append(pts.detach().cpu().reshape(-1, 3))
            pts_n = normalize_func(pts)
            val = query_func(pts_n.reshape(-1, 3)).detach().cpu()
            val = val.reshape(sample_n, z_resolution)
            # print(val.shape)
            u_rays[ui * sample_n:ui * sample_n + len(rays_d), :] = val
            # Fill u_rays here
    pts_volume = torch.cat(pts_set, dim=0).reshape(wid, hei, z_resolution, 3).permute(1, 0, 2, 3)
    u = u_rays.reshape([wid, hei, z_resolution]).permute(1, 0, 2)
    u = u.numpy()

    # u = np.zeros([hei, wid, z_resolution], dtype=np.float32)
    # pts_set = []
    # with torch.no_grad():
    #     for xi, xs in enumerate(X):
    #         for yi, ys in enumerate(Y):
    #             for zi, zs in enumerate(Z):
    #                 xx, yy, zz = torch.meshgrid(xs, ys, zs)
    #                 pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(
    #                     bound_min.device)
    #                 pts_set.append(pts.detach().cpu())
    #                 pts = normalize_func(pts)
    #                 val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
    #                 u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    # pts_set = torch.cat(pts_set, dim=0)

    # u, pts_set = extract_fields(normalize_func, vol_bound, resolution, query_func)
    min_z_idx = np.argmin(np.abs(u), axis=2).reshape([hei, wid, 1, 1]).repeat(3, axis=3)
    pts_select = np.take_along_axis(pts_volume, min_z_idx, axis=2)

    vertices, triangles = mcubes.marching_cubes(u, threshold)
    # bound_min, bound_max = vol_bound
    # b_max_np = bound_max.detach().cpu().numpy()
    # b_min_np = bound_min.detach().cpu().numpy()
    # vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles, pts_volume.reshape(-1, 3), pts_select.reshape(-1, 3)

