# -*- coding: utf-8 -*-
# @Description:
#   Empty worker for new workers writing. This is an example.

# - Package Imports - #
import torch
import torch.nn.functional as F
import numpy as np
from configparser import ConfigParser

import trimesh
from tqdm import tqdm

from worker.worker import Worker
import pointerlib as plb

# from dataset.multi_pat_dataset import MultiPatDataset
from dataset.multi_pat_dataset import PatchLCNDataset
from loss.loss_func import SuperviseLCNLoss, NaiveLoss, SuperviseBCEMaskLoss

from networks.layers import WarpFromXyz
from networks.renderer import NeuSRendererPatch
from networks.neus.fields import SDFNetwork, SingleVarianceNetwork

# from networks.neus import SDFNetwork, SingleVarianceNetwork, NeuSLRenderer


# - Coding Part - #
class ExpXyz2SdfOneShotWorker(Worker):
    def __init__(self, args):
        """
            Please add all parameters will be used in the init function.
        """
        super().__init__(args)
        
        # Fix for NeRF-like training.
        self.N = 1  
        self.sample_num = self.args.batch_num

        # init_dataset()
        self.focus_len = None
        self.pat_dataset = None
        self.warp_layer = None

        # init_networks()
        self.renderer = None

        # init_losses()
        # self.super_loss = None
        self.igr_weight = 0.1
        self.mask_weight = 0.1
        self.lambda_set = []
        for pair_str in args.lambda_stone.split(','):
            epoch_idx, value = pair_str.split('-')
            self.lambda_set.append([int(epoch_idx), float(value)])
        self.warp_lambda = 1.0

        self.anneal_ratio = 1.0
        self.anneal_end = 0

        self.for_viz = {}

    def init_dataset(self):
        """
            Requires:
                self.train_dataset
                self.test_dataset
        """
        config = ConfigParser()
        config.read(str(self.train_dir / 'config.ini'), encoding='utf-8')

        pat_idx_set = [x.strip() for x in self.args.pat_set.split(',')]
        # ref_img_set = [x.strip() for x in self.args.reflect_set.split(',')]
        self.focus_len = plb.str2array(config['RawCalib']['img_intrin'], np.float32)[0]
        self.pat_dataset = PatchLCNDataset(
            scene_folder=self.train_dir,
            pat_idx_set=pat_idx_set,
            sample_num=self.sample_num,
            calib_para=config['RawCalib'],
            device=self.device,
            rad=7
        )
        self.train_dataset = self.pat_dataset

        self.test_dataset = []
        if self.args.exp_type == 'eval':
            self.test_dataset.append(self.pat_dataset)

        self.logging(f'Train_dir: {self.train_dir}')

    def init_networks(self):
        """
            Requires:
                self.networks (dict.)
            Keys will be used for network saving.
        """
        self.networks['SDFNetwork'] = SDFNetwork(
            d_out=1,
            d_in=3,
            d_hidden=256,
            n_layers=8,
            skip_in=[4],
            multires=self.args.multires,  # 6
            bias=0.5,
            scale=1.0,
            geometric_init=True,
            weight_norm=True
        )
        self.networks['SingleVariance'] = SingleVarianceNetwork(
            init_val=0.3
        )
        config = ConfigParser()
        config.read(str(self.train_dir / 'config.ini'), encoding='utf-8')
        self.warp_layer = WarpFromXyz(
            calib_para=config['RawCalib'],
            pat_mat=self.pat_dataset.get_pat_set(),
            scale_mat=self.pat_dataset.get_scale_mat(),
            device=self.device
        )
        self.renderer = NeuSRendererPatch(
            sdf_network=self.networks['SDFNetwork'],
            deviation_network=self.networks['SingleVariance'],
            color_network=self.warp_layer
        )
        self.logging(f'Networks: {",".join(self.networks.keys())}')
        self.logging(f'Networks-static: {",".join(self.network_static_list)}')

    def init_losses(self):
        """
            Requires:
                self.loss_funcs (dict.)
            Keys will be used for avg_meter.
        """
        self.loss_funcs['color_fine'] = SuperviseLCNLoss(dist='l1')
        self.loss_funcs['eikonal'] = NaiveLoss()
        self.loss_funcs['mask_loss'] = SuperviseBCEMaskLoss()
        self.loss_funcs['warp_loss'] = SuperviseLCNLoss(dist='l1')
        self.logging(f'Loss types: {self.loss_funcs.keys()}')
        pass

    def data_process(self, idx, data):
        """
            Process data and move data to self.device.
            output will be passed to :net_forward().
        """
        for key in data:
            data[key] = data[key][0]  # Remove batch wrapper.
        return data

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.n_iter / self.anneal_end])

    def get_sig_anneal_ratio(self):
        # [0.0, 1.0] -> [10.0, 1.0]
        return 8.0  # Fix anneal ratio.

        if self.args.sigma_anneal_end == 0:
            return 1.0
        else:
            scale_step = np.min([1.0, self.n_iter / float(self.args.sigma_anneal_end)])
            return np.clip((1.0 - scale_step) * 9.0 + 1.0, 1.0, 10.0)

    def net_forward(self, data):
        """
            How networks process input data and give out network output.
            The output will be passed to :loss_forward().
        """

        near, far = self.pat_dataset.near_far_from_sphere(data['rays_o'], data['rays_v'])
        render_out = self.renderer.render(
            rays_o=data['rays_o'],
            rays_d=data['rays_v'],
            rays_d_patch=data['rays_v_patch'],
            reflect=data['reflect'],
            near=near,
            far=far,
            cos_anneal_ratio=self.get_cos_anneal_ratio(),
            patch_render=True
        )
        return render_out

    def loss_forward(self, net_out, data):
        """
            How loss functions process the output from network and input data.
            The output will be used with err.backward().
            data: {
                'rays_o':       [N, 3],
                'rays_v':       [N, 3],
                'rays_v_patch': [N, 3 * pch_num],
                'mask':         [N, 1],
                'maks_patch':   [N, pch_num],
                'color':        [N, C],
                'color_patch':  [N, C * pch_num],
                'mu':           [N, 1],
                'std':          [N, 1]
            }
            net_out: {
                'color_fine': [N, C * pch_num]
                'gradients': [N, sample, 3],
                'weights': [N, sample],
                'gradient_error': [1],
                'inside_sphere': ret_fine['inside_sphere']
            }
        """
        total_loss = torch.zeros(1).to(self.device)

        # color_fine_loss
        kernel = self.pat_dataset.get_kernel(self.get_sig_anneal_ratio()).to(self.device)  # [1, pch_num]
        color_fine_loss, mask = self.loss_record(
            'color_fine', pred=net_out['color_fine'], 
            target=data['color_patch'], mask=data['mask_patch'] * kernel,
            return_val_only=False,
        )
        total_loss += color_fine_loss
        self.for_viz['color_pred'] = net_out['color_fine']
        self.for_viz['color_target'] = data['color_patch']
        self.for_viz['mask_patch'] = mask
        self.for_viz['color'] = data['color']

        # eikonal_loss
        eikonal_loss = self.loss_record(
            'eikonal', loss=net_out['gradient_error']
        )
        total_loss += eikonal_loss * self.igr_weight
        
        # mask_loss
        mask_loss = self.loss_record(
            'mask_loss', pred=net_out['weight_sum'].clip(1e-3, 1.0 - 1e-3),
            target=data['mask']
        )
        total_loss += mask_loss * self.mask_weight

        # warp_loss
        # warp_loss = self.loss_record(
        #     'warp_loss', pred=net_out['color_1pt'],
        #     target=data['color'], mask=data['mask']
        # )
        # total_loss += warp_loss * self.warp_lambda

        self.avg_meters['Total'].update(total_loss, self.N)
        return total_loss

    def callback_after_train(self, epoch):
        # anneal_ratio
        # self.anneal_ratio = np.min([1.0, epoch / self.anneal_end])
        for lambda_epoch, lambda_value in self.lambda_set:
            if lambda_epoch > epoch:
                break
            self.warp_lambda = lambda_value
        pass

    def check_save_res(self, epoch):
        """
            Rewrite the report.
        """
        return False

    def check_realtime_report(self, **kwargs):
        """
            Rewrite the report.
        """
        return False

    def callback_epoch_report(self, epoch, tag, stopwatch):
        """
            Rewrite the report.
        """
        total_loss = self.avg_meters['Total'].get_epoch()
        for name in self.avg_meters.keys():
            self.loss_writer.add_scalar(f'{tag}-epoch/{name}', self.avg_meters[name].get_epoch(), epoch)
            self.avg_meters[name].clear_epoch()

        if epoch % self.args.report_stone == 0:
            self.logging(f'Timings: {stopwatch}, total_loss={total_loss}', tag=tag, step=epoch)

        # Markdown best performance
        target_tag = 'eval' if not isinstance(self.test_dataset, list) else 'eval0'  # First
        if tag == target_tag:
            if self.history_best[0] is None or self.history_best[0] > total_loss:
                self.history_best = [total_loss, True]

        # For image
        if self.args.debug_mode or (self.args.img_stone > 0 and epoch % self.args.img_stone == 0):
            self.callback_img_visual(None, None, tag, epoch)

        # For save
        if self.args.save_stone > 0 and epoch % self.args.save_stone == 0:
            self.callback_save_res(epoch, None, None, self.pat_dataset)

        self.loss_writer.flush()

    def check_img_visual(self, **kwargs):
        """
            Rewrite the report.
        """
        return False

    def callback_img_visual(self, data, net_out, tag, step):
        """
            The callback function for visualization.
            Notice that the loss will be automatically report and record to writer.
            For image visualization, some custom code is needed.
            Please write image to loss_writer and call flush().
        """
        res = self.visualize_output(resolution_level=4, require_item=['img_gen', 'norm', 'mesh'])
        self.loss_writer.add_image(f'{tag}/img_render', res['img_gen'], step)
        self.loss_writer.add_image(f'{tag}/img_norm', res['norm'], step)
        vertices, triangles = res['mesh']
        vertices = torch.from_numpy(vertices.astype(np.float32)).unsqueeze(0)
        triangles = torch.from_numpy(triangles.astype(np.int32)).unsqueeze(0)
        self.loss_writer.add_mesh(f'{tag}/mesh', vertices=vertices, faces=triangles, global_step=step)

        # Images for computing
        batch, ch = self.for_viz['color'].shape
        pch_num = self.for_viz['mask_patch'].shape[1]
        h_patch = 2 ** int(np.log2(batch) // 2)
        w_patch = int(batch // h_patch)
        side_len = int(np.sqrt(pch_num))

        def reshape_patches(input_patches, c):
            viz_res = input_patches.detach().cpu().reshape(h_patch, w_patch, c, side_len, side_len)
            viz_res = viz_res.permute(0, 2, 3, 1, 4).reshape(h_patch * c * side_len, w_patch * side_len)
            viz_res = plb.VisualFactory.img_visual(viz_res)
            return viz_res

        color_pred = reshape_patches(self.for_viz['color_pred'], ch)
        color_target = reshape_patches(self.for_viz['color_target'], ch)
        mask_patch = self.for_viz['mask_patch'].detach().cpu()
        mask_patch /= mask_patch.max()
        mask_patch = reshape_patches(mask_patch, 1)
        self.loss_writer.add_image(f'{tag}/pred_patch', color_pred, step)
        self.loss_writer.add_image(f'{tag}/target_patch', color_target, step)
        self.loss_writer.add_image(f'{tag}/mask_patch', mask_patch, step)

        self.loss_writer.flush()

    def callback_save_res(self, epoch, data, net_out, dataset):
        """
            The callback function for data saving.
            The data should be saved with the input.
            Please create new folders and save result.
        """
        res = self.visualize_output(resolution_level=1, require_item=['img_gen', 'norm', 'mesh'])
        save_folder = self.res_dir / 'output' / f'e_{epoch:05}'
        save_folder.mkdir(parents=True, exist_ok=True)
        plb.imsave(save_folder / 'wrp_viz.png', res['img_gen'])
        img_folder = save_folder / 'img'
        for i, img in enumerate(res['render_list']):
            plb.imsave(img_folder / f'img_{i}.png', img, mkdir=True)
        plb.imsave(save_folder / 'norm.png', res['norm'])
        vertices, triangles = res['mesh']
        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(str(save_folder / 'mesh.ply'))
        pass

    def visualize_output(self, resolution_level, require_item=None):
        if require_item is None:
            require_item = [
                'img_gen',
                'norm',
                'mesh',
            ]

        pvf = plb.VisualFactory
        def require_contain(*keys):
            flag = False
            for key in keys:
                flag = flag or (key in require_item)
            return flag

        res = {}
        if require_contain('img_gen', 'norm'):
            # image
            rays_o, rays_v, reflect, mask = self.pat_dataset.gen_uniform_ray(
                resolution_level=resolution_level
            )
            hei, wid = rays_o.shape[-2:]

            rays_o = rays_o.reshape(rays_o.shape[0], -1).permute(1, 0)
            rays_v = rays_v.reshape(rays_v.shape[0], -1).permute(1, 0)
            reflect = reflect.reshape(reflect.shape[0], -1).permute(1, 0)
            mask = mask.reshape(mask.shape[0], -1).permute(1, 0)

            sample_num = self.sample_num * 16
            rays_o_set = rays_o.split(sample_num)
            rays_v_set = rays_v.split(sample_num)
            reflect_set = reflect.split(sample_num)
            mask_set = mask.split(sample_num)

            out_rgb_fine = []
            out_rgb_1pt = []
            out_norm = []
            pbar = tqdm(total=len(rays_o_set), desc=f'Vis-Resolution={resolution_level}')
            for rays_o, rays_v, reflect in zip(rays_o_set, rays_v_set, reflect_set):
                pbar.update(1)
                near, far = self.pat_dataset.near_far_from_sphere(rays_o, rays_v)
                render_out = self.renderer.render(
                    rays_o=rays_o,
                    rays_d=rays_v,
                    rays_d_patch=None,
                    near=near,
                    far=far,
                    cos_anneal_ratio=self.anneal_ratio,
                    patch_render=False
                )
                out_rgb_fine.append(render_out['color_fine'].detach().cpu())
                out_rgb_1pt.append(render_out['color_1pt'].detach().cpu())
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu()
                out_norm.append(normals)
                del render_out
            pbar.close()

            img_render_list = []
            if len(out_rgb_fine) > 0:
                img_fine = torch.cat(out_rgb_fine, dim=0).permute(1, 0)  # [C, N]
                img_fine = img_fine.reshape(-1, hei, wid)
                for c in range(img_fine.shape[0]):
                    img_viz = pvf.img_visual(img_fine[c])
                    img_render_list.append(img_viz)

            img_warp_list = []
            if len(out_rgb_1pt) > 0:
                img_1pt = torch.cat(out_rgb_1pt, dim=0).permute(1, 0)  # [C, N]
                img_1pt = img_1pt.reshape(-1, hei, wid)
                for c in range(img_1pt.shape[0]):
                    img_viz = pvf.img_visual(img_1pt[c])
                    img_warp_list.append(img_viz)

            img_gt_list = []
            img_gt_set = self.pat_dataset.get_cut_img(resolution_level)
            for c in range(img_gt_set.shape[0]):
                img_viz = pvf.img_visual(img_gt_set[c].detach().cpu())
                img_gt_list.append(img_viz)

            if len(out_norm) > 0:
                normal_img = torch.cat(out_norm, dim=0).permute(1, 0)  # [C, N]
                normal_img = normal_img.reshape(-1, hei, wid)
                normal_img_uni = (normal_img + 1.0) / 2.0
                res['norm'] = normal_img_uni

            wrp_viz = pvf.img_concat(img_warp_list + img_render_list + img_gt_list,
                                     3, len(img_render_list), transpose=False)
            res['render_list'] = img_render_list
            res['img_gen'] = wrp_viz

        if require_contain('mesh'):
            bound_min = torch.tensor(self.pat_dataset.object_bbox_min, dtype=torch.float32)
            bound_max = torch.tensor(self.pat_dataset.object_bbox_max, dtype=torch.float32)

            vertices, triangles = self.renderer.extract_geometry(
                bound_min[:3], bound_max[:3], resolution=512 // resolution_level, threshold=0.0
            )
            scale_mat = self.pat_dataset.get_scale_mat()
            vertices = vertices * scale_mat[0, 0] + scale_mat[:3, 3][None]
            res['mesh'] = (vertices, triangles)

        return res
