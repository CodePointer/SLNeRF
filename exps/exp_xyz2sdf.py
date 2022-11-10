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
from dataset.multi_pat_dataset import MultiPatDataset

from loss.loss_func import SuperviseDistLoss, NaiveLoss
from networks.layers import WarpFromXyz
from networks.neus import SDFNetwork, SingleVarianceNetwork, NeuSLRenderer


# - Coding Part - #
class ExpXyz2SdfWorker(Worker):
    def __init__(self, args):
        """
            Please add all parameters will be used in the init function.
        """
        super().__init__(args)

        # init_dataset()
        self.focus_len = None
        self.pat_dataset = None
        self.sample_num = self.args.batch_num
        self.bound = None
        self.warp_layer = None

        # init_networks()
        self.renderer = None

        # init_losses()
        self.super_loss = None

        self.anneal_ratio = 1.0
        self.anneal_end = 50000

    def init_dataset(self):
        """
            Requires:
                self.train_dataset
                self.test_dataset
                self.res_writers  (self.create_res_writers)
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

        self.res_writers = []
        if self.args.save_stone > 0:
            self.res_writers.append(self.create_res_writers())

        self.bound = self.pat_dataset.get_bound()
        self.logging(f'--train_dir: {self.train_dir}')

    def init_networks(self):
        """
            Requires:
                self.networks (dict.)
            Keys will be used for network saving.
        """
        self.networks['SDFNetwork'] = SDFNetwork(
            d_out=257,
            d_in=3,
            d_hidden=256,
            n_layers=8,
            skip_in=[4],
            multires=self.args.multires,
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
            pat_mat=self.pat_dataset.pat_set,
            bound=self.bound,
            device=self.device
        )
        self.renderer = NeuSLRenderer(
            sdf_network=self.networks['SDFNetwork'],
            deviation_network=self.networks['SingleVariance'],
            color_network=self.warp_layer,
            n_samples=64,
            n_importance=64,
            up_sample_steps=4,
            perturb=1.0,
            bound=self.bound
        )
        self.logging(f'--networks: {",".join(self.networks.keys())}')
        self.logging(f'--networks-static: {",".join(self.network_static_list)}')

    def init_losses(self):
        """
            Requires:
                self.loss_funcs (dict.)
            Keys will be used for avg_meter.
        """
        self.super_loss = SuperviseDistLoss(dist='l1')
        self.loss_funcs['color_l1'] = self.super_loss
        self.loss_funcs['eikonal'] = NaiveLoss()
        self.logging(f'--loss types: {self.loss_funcs.keys()}')
        pass

    def data_process(self, idx, data):
        """
            Process data and move data to self.device.
            output will be passed to :net_forward().
        """
        for key in data:
            data[key] = data[key][0]
        return data

    def get_cos_anneal_ratio(self):
        return self.anneal_ratio

    def net_forward(self, data):
        """
            How networks process input data and give out network output.
            The output will be passed to :loss_forward().
        """

        render_out = self.renderer.render(
            rays_o=data['rays_o'],
            rays_d=data['rays_v'],
            bound=self.bound,
            reflect=data['reflect'],
            background_rgb=None,
            cos_anneal_ratio=self.anneal_ratio
        )
        return render_out['color_fine'], render_out['gradient_error']

    def loss_forward(self, net_out, data):
        """
            How loss functions process the output from network and input data.
            The output will be used with err.backward().
        """
        color_fine, gradient_loss = net_out
        total_loss = torch.zeros(1).to(self.device)

        total_loss += self.loss_record(
            'color_l1', pred=color_fine, target=data['color']
        )

        total_loss += self.loss_record(
            'eikonal', loss=gradient_loss
        ) * 0.1

        self.avg_meters['Total'].update(total_loss, self.N)
        return total_loss

    def callback_after_train(self, epoch):
        # anneal_ratio
        self.anneal_ratio = np.min([1.0, epoch / self.anneal_end])

    def callback_save_res(self, data, net_out, dataset, res_writer):
        """
            The callback function for data saving.
            The data should be saved with the input.
            Please create new folders and save result.
        """
        # out_dir, config = res_writer
        # for i in range(len(net_out)):
        #     plb.imsave(out_dir / f'depth_s{i}.png', net_out[i], scale=10.0, img_type=np.uint16)
        pass

    def check_realtime_report(self, **kwargs):
        """
            Rewrite the report.
        """
        return False

    def check_img_visual(self, **kwargs):
        """
            The img_visual callback entries.
            Modified this function can control the report strategy during training and testing.
            (Additional)
        """
        epoch = kwargs['epoch']
        if self.args.debug_mode:
            return True
        if epoch % self.args.img_stone == 0:
            return True
        return False
        # return super().check_img_visual(**kwargs)

    def callback_epoch_report(self, epoch, tag, stopwatch, res_writer=None):
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

        # Save
        if self.args.save_stone > 0 and epoch % self.args.save_stone == 0:
            res = self.visualize_output(resolution_level=1, require_item=[
                'wrp_viz', 'mesh'
            ])  # TODO: 可视化所有的query部分，用于绘制结果图。
            save_folder = self.res_dir / 'output' / f'e_{epoch:05}'
            save_folder.mkdir(parents=True, exist_ok=True)
            plb.imsave(save_folder / f'wrp_viz.png', res['wrp_viz'])
            vertices, triangles = res['mesh']
            mesh = trimesh.Trimesh(vertices, triangles)
            mesh.export(str(save_folder / f'mesh.ply'))

        self.loss_writer.flush()

    def visualize_output(self, resolution_level, require_item=None):
        if require_item is None:
            require_item = [
                'wrp_viz',
                'mesh',
            ]

        pvf = plb.VisualFactory
        def require_contain(*keys):
            flag = False
            for key in keys:
                flag = flag or (key in require_item)
            return flag

        res = {}
        if require_contain('wrp_viz'):
            # image
            rays_o, rays_d, img_size, mask_val, reflect = self.pat_dataset.get_uniform_ray(
                resolution_level=resolution_level
            )

            img_hei, img_wid = img_size
            total_ray = rays_o.shape[0]
            idx = torch.arange(0, total_ray, dtype=torch.long)
            idx = idx[mask_val > 0.0]
            rays_o = rays_o[mask_val > 0.0]
            rays_d = rays_d[mask_val > 0.0]
            reflect = reflect[mask_val > 0.0]

            pch_num = self.pat_dataset.get_pch_len() ** 2
            rays_o_set = rays_o.split(self.sample_num * pch_num)
            rays_d_set = rays_d.split(self.sample_num * pch_num)
            reflect_set = reflect.split(self.sample_num * pch_num)

            out_rgb_fine = []
            for rays_o, rays_d, reflect in zip(rays_o_set, rays_d_set, reflect_set):
                render_out = self.renderer.render(rays_o, rays_d, self.bound, reflect,
                                                  cos_anneal_ratio=self.anneal_ratio)
                color_fine = render_out['color_fine']  # [N, C]
                out_rgb_fine.append(color_fine.detach().cpu())
                del render_out

            img_set = torch.cat(out_rgb_fine, dim=0).permute(1, 0)
            channel = img_set.shape[0]
            img_all = torch.zeros([channel, total_ray], dtype=img_set.dtype).to(img_set.device)
            img_all.scatter_(1, idx.reshape(1, -1).repeat(channel, 1), img_set)
            img_render_list = []
            for c in range(channel):
                img_fine = img_all[c].reshape(img_wid, img_hei).permute(1, 0)
                img_viz = pvf.img_visual(img_fine)
                img_render_list.append(img_viz)

            img_gt_list = []
            img_input = self.pat_dataset.img_set.unsqueeze(0).detach().cpu()
            img_input_re = torch.nn.functional.interpolate(img_input, scale_factor=1 / resolution_level).squeeze()
            for c in range(channel):
                img_gt = img_input_re[c]
                img_viz = pvf.img_visual(img_gt)
                img_render_list.append(img_viz)

            wrp_viz = pvf.img_concat(img_render_list + img_gt_list, 2, channel, transpose=False)
            res['wrp_viz'] = wrp_viz

        if require_contain('mesh'):
            vertices, triangles = self.renderer.extract_geometry(
                vol_bound=self.bound,
                resolution=64,  # 256 // resolution_level
            )
            res['mesh'] = (vertices, triangles)

        return res

    def callback_img_visual(self, data, net_out, tag, step):
        """
            The callback function for visualization.
            Notice that the loss will be automatically report and record to writer.
            For image visualization, some custom code is needed.
            Please write image to loss_writer and call flush().
        """
        res = self.visualize_output(resolution_level=4, require_item=['wrp_viz', 'mesh'])
        self.loss_writer.add_image(f'{tag}/img_render', res['wrp_viz'], step)
        vertices, triangles = res['mesh']
        vertices = torch.from_numpy(vertices.astype(np.float32)).unsqueeze(0)
        triangles = torch.from_numpy(triangles.astype(np.int32)).unsqueeze(0)
        self.loss_writer.add_mesh(f'{tag}/mesh', vertices=vertices, faces=triangles, global_step=step)
        self.loss_writer.flush()
