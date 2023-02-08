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
from dataset.dataset_neus import MultiPatDataset
from loss.loss_func import SuperviseDistLoss, NaiveLoss, SuperviseBCEMaskLoss

from networks.layers import WarpFromXyz
from networks.neus.renderer import NeuSRenderer
from networks.neus.fields import SDFNetwork, SingleVarianceNetwork

# from networks.neus import SDFNetwork, SingleVarianceNetwork, NeuSLRenderer


# - Coding Part - #
class ExpXyz2SdfWorker(Worker):
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

        self.anneal_ratio = 1.0
        self.anneal_end = 0

    def init_dataset(self):
        """
            Requires:
                self.train_dataset
                self.test_dataset
        """
        config = ConfigParser()
        config.read(str(self.train_dir / 'config.ini'), encoding='utf-8')

        pat_idx_set = [int(x.strip()) for x in self.args.pat_set.split(',')]
        ref_img_set = [int(x.strip()) for x in self.args.reflect_set.split(',')]
        self.focus_len = plb.str2array(config['Calibration']['img_intrin'], np.float32)[0]
        self.pat_dataset = MultiPatDataset(
            scene_folder=self.train_dir,
            pat_idx_set=pat_idx_set,
            ref_img_set=ref_img_set,
            sample_num=self.sample_num,
            calib_para=config['Calibration'],
            device=self.device
        )
        self.train_dataset = self.pat_dataset

        self.test_dataset = []
        if self.args.exp_type == 'eval':
            self.test_dataset.append(self.pat_dataset)

        self.logging(f'--train_dir: {self.train_dir}')

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
            calib_para=config['Calibration'],
            pat_mat=self.pat_dataset.get_pat_set(),
            scale_mat=self.pat_dataset.get_scale_mat(),
            device=self.device
        )
        self.renderer = NeuSRenderer(
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
        self.loss_funcs['color_fine'] = SuperviseDistLoss(dist='l1')
        self.loss_funcs['eikonal'] = NaiveLoss()
        self.loss_funcs['mask_loss'] = SuperviseBCEMaskLoss()
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
            return np.min([1.0, self.iter_step / self.anneal_end])

    def net_forward(self, data):
        """
            How networks process input data and give out network output.
            The output will be passed to :loss_forward().
        """

        near, far = self.pat_dataset.near_far_from_sphere(data['rays_o'], data['rays_v'])
        background_rgb = None
        render_out = self.renderer.render(
            rays_o=data['rays_o'],
            rays_d=data['rays_v'],
            reflect=data['reflect'],
            near=near,
            far=far,
            background_rgb=background_rgb,
            cos_anneal_ratio=self.get_cos_anneal_ratio()
        )
        return render_out

    def loss_forward(self, net_out, data):
        """
            How loss functions process the output from network and input data.
            The output will be used with err.backward().
            data: {
                'rays_o': self.rays_o,                          # [N, 3]
                'rays_v': rays_v,                               # [N, 3]
                'mask': mask,                                   # [N, 1]
                'color': color,                                 # [N, C]
                'reflect': ref,                                 # [N, 2]
            }
            net_out: {
                'color_fine': color_fine,
                's_val': s_val,
                'cdf_fine': ret_fine['cdf'],
                'weight_sum': weights_sum,
                'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
                'gradients': gradients,
                'weights': weights,
                'gradient_error': ret_fine['gradient_error'],
                'inside_sphere': ret_fine['inside_sphere']
            }
        """
        total_loss = torch.zeros(1).to(self.device)

        # color_fine_loss
        color_fine_loss = self.loss_record(
            'color_fine', pred=net_out['color_fine'], 
            target=data['color'], mask=data['mask'],
        )
        total_loss += color_fine_loss

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

        self.avg_meters['Total'].update(total_loss, self.N)
        return total_loss

    def callback_after_train(self, epoch):
        # anneal_ratio
        # self.anneal_ratio = np.min([1.0, epoch / self.anneal_end])
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
        if self.args.debug_mode or epoch % self.args.img_stone == 0:
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

            rays_o_set = rays_o.split(self.sample_num)
            rays_v_set = rays_v.split(self.sample_num)
            reflect_set = reflect.split(self.sample_num)
            mask_set = mask.split(self.sample_num)

            out_rgb_fine = []
            out_norm = []
            for rays_o, rays_v, reflect in zip(rays_o_set, rays_v_set, reflect_set):
                near, far = self.pat_dataset.near_far_from_sphere(rays_o, rays_v)
                render_out = self.renderer.render(
                    rays_o, rays_v, reflect, near, far,
                    cos_anneal_ratio=self.anneal_ratio
                )
                out_rgb_fine.append(render_out['color_fine'].detach().cpu())
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu()
                out_norm.append(normals)
                del render_out

            img_render_list = []
            if len(out_rgb_fine) > 0:
                img_fine = torch.cat(out_rgb_fine, dim=0).permute(1, 0)  # [C, N]
                img_fine = img_fine.reshape(-1, hei, wid)
                for c in range(img_fine.shape[0]):
                    img_viz = pvf.img_visual(img_fine[c])
                    img_render_list.append(img_viz)

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

            wrp_viz = pvf.img_concat(img_render_list + img_gt_list, 2, len(img_render_list), transpose=False)
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
