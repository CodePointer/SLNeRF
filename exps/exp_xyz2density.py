# -*- coding: utf-8 -*-
# @Description:
#   Empty worker for new workers writing. This is an example.

# - Package Imports - #
import torch
import torch.nn.functional as F
import numpy as np
from configparser import ConfigParser
import matplotlib.pyplot as plt

from worker.worker import Worker
import pointerlib as plb
from dataset.multi_pat_dataset import MultiPatDataset

from loss.loss_func import SuperviseDistLoss, NeighborGradientLossWithEdge
from networks.layers import WarpFromXyz
from networks.neus import NeuSLRenderer, DensityNetwork, ReflectNetwork


# - Coding Part - #
class ExpXyz2DensityWorker(Worker):
    def __init__(self, args):
        """
            Please add all parameters will be used in the init function.
        """
        super().__init__(args)

        # init_dataset()
        self.pat_dataset = None
        self.sample_num = self.args.batch_num
        self.bound = None
        self.warp_layer = None

        # init_networks()
        self.renderer = None

        # init_losses()
        self.super_loss = None
        self.igr_weight = 0.1

        self.anneal_ratio = 1.0
        self.anneal_end = 50000

        self.alpha_set = []
        for pair_str in args.alpha_stone.split(','):
            epoch_idx, value = pair_str.split('-')
            self.alpha_set.append([int(epoch_idx), float(value)])
        self.alpha = self.alpha_set[0][1]

        self.reg_set = []
        for pair_str in args.reg_stone.split(','):
            epoch_idx, value = pair_str.split('-')
            self.reg_set.append([int(epoch_idx), float(value)])
        self.reg_ratio = self.reg_set[0][1]


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
        self.pat_dataset = MultiPatDataset(
            scene_folder=self.train_dir,
            pat_idx_set=pat_idx_set,
            sample_num=self.sample_num,
            calib_para=config['Calibration'],
            device=self.device,
            rad=self.args.patch_rad
        )
        self.train_dataset = self.pat_dataset

        self.test_dataset = []
        if self.args.exp_type == 'eval':
            self.test_dataset.append(self.pat_dataset)

        self.res_writers = []
        if self.args.save_stone > 0:
            self.res_writers.append(self.create_res_writers())

        # Init warp_layer
        self.bound = self.pat_dataset.get_bound()
        self.warp_layer = WarpFromXyz(
            calib_para=config['Calibration'],
            pat_mat=self.pat_dataset.pat_set,
            bound=self.bound,
            device=self.device
        )

        self.logging(f'--train_dir: {self.train_dir}')

    def init_networks(self):
        """
            Requires:
                self.networks (dict.)
            Keys will be used for network saving.
        """
        self.networks['DensityNetwork'] = DensityNetwork(
            d_out=1 + 256,
            d_in=3,
            d_hidden=256,
            n_layers=8,
            skip_in=[4],
            multires=6,
            bias=0.5,
            scale=1.0,
            geometric_init=True,
            weight_norm=True
        )
        self.networks['ReflectNetwork'] = ReflectNetwork(
            d_feature=256,
            d_in=3,  # Points only
            d_out=2,  # Reflectence: a, b
            d_hidden=256,
            n_layers=4,
            warp_layer=self.warp_layer,
            weight_norm=True,
            multires_view=4,
            squeeze_out=True
        )
        self.renderer = NeuSLRenderer(
            sdf_network=self.networks['DensityNetwork'],
            deviation_network=None,
            color_network=self.networks['ReflectNetwork'],
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

        if self.args.patch_rad > 0:
            self.loss_funcs['gradient'] = NeighborGradientLossWithEdge(
                rad=self.args.patch_rad,
                dist='l2',
                sigma=self.args.reg_color_sigma
            )

        self.logging(f'--loss types: {self.loss_funcs.keys()}')
        pass

    def data_process(self, idx, data):
        """
            Process data and move data to self.device.
            output will be passed to :net_forward().
        """
        for key in data:
            data[key] = data[key][0]

        data['reflect'] = data['color'][:, -2:]
        return data

    def get_cos_anneal_ratio(self):
        return self.anneal_ratio

    def net_forward(self, data):
        """
            How networks process input data and give out network output.
            The output will be passed to :loss_forward().
        """
        render_out = self.renderer.render_density(data['rays_v'], reflect=data['reflect'], alpha=self.alpha)
        return render_out['color'], render_out['depth']

    def loss_forward(self, net_out, data):
        """
            How loss functions process the output from network and input data.
            The output will be used with err.backward().
        """
        color_fine, depth_res = net_out
        total_loss = torch.zeros(1).to(self.device)
        total_loss += self.loss_record(
            'color_l1', pred=color_fine, target=data['color']
        )

        if 'gradient' in self.loss_funcs:
            total_loss += self.loss_record(
                'gradient', depth=depth_res, mask=data['mask'], color=data['color']
            ) * self.reg_ratio

        self.avg_meters['Total'].update(total_loss, self.N)
        return total_loss

    def callback_after_train(self, epoch):
        # anneal_ratio
        self.anneal_ratio = np.min([1.0, epoch / self.anneal_end])
        for alpha_pair in self.alpha_set:
            if alpha_pair[0] > epoch:
                break
            self.alpha = alpha_pair[1]
        for reg_epoch, reg_value in self.reg_set:
            if reg_epoch > epoch:
                break
            self.reg_ratio = reg_value

    def callback_save_res(self, data, net_out, dataset, res_writer):
        """
            The callback function for data saving.
            The data should be saved with the input.
            Please create new folders and save result.
        """
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
                'img_list', 'wrp_viz', 'depth_viz', 'depth_map', 'point_cloud'
            ])  # TODO: 可视化所有的query部分，用于绘制结果图。
            save_folder = self.res_dir / 'output' / f'e_{epoch:05}'
            save_folder.mkdir(parents=True, exist_ok=True)
            plb.imsave(save_folder / f'wrp_viz.png', res['wrp_viz'])
            plb.imsave(save_folder / f'depth_viz.png', res['depth_viz'])
            plb.imsave(save_folder / f'depth_map.png', res['depth_map'], scale=10.0, img_type=np.uint16)
            np.savetxt(str(save_folder / f'depth_map.asc'), res['point_cloud'],
                       fmt='%.2f', delimiter=',', newline='\n', encoding='utf-8')
            img_folder = save_folder / 'img'
            for i, img in enumerate(res['img_list']):
                plb.imsave(img_folder / f'img_{i}.png', img, mkdir=True)

        pass

        self.loss_writer.flush()

    def visualize_output(self, resolution_level, require_item=None):
        if require_item is None:
            require_item = [
                # 'img_list',
                # 'reflectance',
                'wrp_viz',

                # 'depth_map',
                'depth_viz',
                'point_cloud',
                'mesh',

                # 'query_points',
                # 'query_z',
                # 'query_color',
                # 'query_density',
                # 'query_weights',
            ]

        pvf = plb.VisualFactory

        def require_contain(*keys):
            flag = False
            for key in keys:
                flag = flag or (key in require_item)
            return flag

        # image, depth_map, point_cloud
        rays_o, rays_d, img_size, mask_val, reflect = self.pat_dataset.get_uniform_ray(resolution_level=resolution_level)

        img_hei, img_wid = img_size
        total_ray = rays_o.shape[0]
        idx = torch.arange(0, total_ray, dtype=torch.long)
        idx = idx[mask_val > 0.0]
        rays_o = rays_o[mask_val > 0.0]
        rays_d = rays_d[mask_val > 0.0]
        reflect = reflect[mask_val > 0.0]

        pch_num = (self.args.patch_rad * 2 + 1) ** 2
        rays_o_set = rays_o.split(self.sample_num * pch_num)
        rays_d_set = rays_d.split(self.sample_num * pch_num)
        reflect_set = reflect.split(self.sample_num * pch_num)
        out_rgb_fine = []
        out_reflectance = []
        out_depth = []
        out_z = []
        out_weights = []
        out_pts = []
        out_color = []
        out_density = []
        for rays_o, rays_d, reflect in zip(rays_o_set, rays_d_set, reflect_set):
            render_out = self.renderer.render_density(rays_d, alpha=self.alpha, reflect=reflect)

            if require_contain('img_list', 'wrp_viz'):
                color_fine = render_out['color']  # [N, C]
                out_rgb_fine.append(color_fine.detach().cpu())

            if require_contain('reflectance'):
                out_reflectance.append(render_out['reflectance'].detach().cpu())  # [N, 2]

            if require_contain('depth_map', 'depth_viz', 'point_cloud', 'mesh'):
                # weights = render_out['weights']
                # mid_z_vals = render_out['pts'][:, :, -1]
                # max_idx = torch.argmax(weights, dim=1)  # [N]
                # mid_z = mid_z_vals[torch.arange(max_idx.shape[0]), max_idx]
                # out_depth.append(mid_z.detach().cpu())
                depth_val = render_out['depth'].reshape(-1)
                out_depth.append(depth_val.detach().cpu())

            if require_contain('query_z'):
                out_z.append(render_out['z_vals'].detach().cpu())

            if require_contain('query_points'):
                out_pts.append(render_out['pts'].detach().cpu())

            if require_contain('query_color'):
                out_color.append(render_out['pt_color'].detach().cpu())

            if require_contain('query_density'):
                out_density.append(render_out['density'].detach().cpu())

            if require_contain('query_weights'):
                out_weights.append(render_out['weights'].detach().cpu())

        res = {}
        if require_contain('img_list', 'wrp_viz'):
            img_set = torch.cat(out_rgb_fine, dim=0).permute(1, 0)
            channel = img_set.shape[0]
            img_all = torch.zeros([channel, total_ray], dtype=img_set.dtype).to(img_set.device)
            img_all.scatter_(1, idx.reshape(1, -1).repeat(channel, 1), img_set)
            img_render_list = []
            for c in range(channel):
                img_fine = img_all[c].reshape(img_wid, img_hei).permute(1, 0)
                img_viz = pvf.img_visual(img_fine)
                img_render_list.append(img_viz)

            if require_contain('img_list'):
                res['img_list'] = img_render_list

            if require_contain('wrp_viz'):
                img_gt_list = []
                img_input = self.pat_dataset.img_set.unsqueeze(0).detach().cpu()
                img_input_re = torch.nn.functional.interpolate(img_input, scale_factor=1 / resolution_level).squeeze(dim=0)
                for c in range(channel):
                    img_gt = img_input_re[c]
                    img_viz = pvf.img_visual(img_gt)
                    img_gt_list.append(img_viz)
                wrp_viz = pvf.img_concat(img_render_list + img_gt_list, 2, channel, transpose=False)
                res['wrp_viz'] = wrp_viz

        if require_contain('reflectance'):
            ref_set = torch.cat(out_reflectance, dim=0).permute(1, 0)
            ref_all = torch.zeros([2, total_ray], dtype=ref_set.dtype).to(ref_set.device)
            ref_all.scatter_(1, idx.reshape(1, -1).repeat(2, 1), ref_set)
            ref_viz_list = []
            for c in range(2):
                ref_map = ref_all[c].reshape(img_wid, img_hei).permute(1, 0)
                ref_viz_list.append(pvf.img_visual(ref_map))
            res['reflectance'] = ref_viz_list

        if require_contain('depth_map', 'depth_viz', 'point_cloud', 'mesh'):
            depth_set = torch.cat(out_depth, dim=0).reshape(-1)
            depth_all = torch.zeros([total_ray], dtype=depth_set.dtype).to(depth_set.device)
            depth_all.scatter_(0, idx, depth_set)
            depth_map = depth_all.reshape(img_wid, img_hei).permute(1, 0)
            if require_contain('depth_map'):
                res['depth_map'] = depth_map
            if require_contain('depth_viz'):
                depth_viz = pvf.disp_visual(depth_map, range_val=self.bound[:, 2].to(depth_map.device))
                res['depth_viz'] = depth_viz
            if require_contain('point_cloud', 'mesh'):
                # TODO: DepthMapVisual + mask
                visualizer = plb.DepthMapVisual(depth_map.unsqueeze(0), focus=1197.0 / resolution_level)
                if require_contain('point_cloud'):
                    res['point_cloud'] = visualizer.to_xyz_set()
                if require_contain('mesh'):
                    res['mesh'] = visualizer.to_mesh()

        if require_contain('query_z'):
            z_set = torch.cat(out_z, dim=0)  # [N, n_sample]
            n_sample = z_set.shape[1]
            z_all = torch.zeros([total_ray, n_sample], dtype=z_set.dtype).to(z_set.device)
            z_all.scatter_(0, idx.reshape(-1, 1).repeat(1, n_sample), z_set)
            z_map = z_all.reshape(img_wid, img_hei, n_sample).permute(1, 0, 2)
            res['query_z'] = z_map

        if require_contain('query_points'):
            pts_set = torch.cat(out_pts, dim=0)  # [N, n_sample, 3]
            n_sample = pts_set.shape[1]
            pts_all = torch.zeros([total_ray, n_sample, 3], dtype=pts_set.dtype).to(pts_set.device)
            pts_all.scatter_(0, idx.reshape(-1, 1, 1).repeat(1, n_sample, 3), pts_set)
            pts_map = pts_all.reshape(img_wid, img_hei, n_sample, 3).permute(1, 0, 2, 3)
            res['query_points'] = pts_map
        if require_contain('query_density'):
            density_set = torch.cat(out_density, dim=0)  # [N, n_sample]
            n_sample = density_set.shape[1]
            density_all = torch.zeros([total_ray, n_sample], dtype=density_set.dtype).to(density_set.device)
            density_all.scatter_(0, idx.reshape(-1, 1).repeat(1, n_sample), density_set)
            density_map = density_all.reshape(img_wid, img_hei, n_sample).permute(1, 0, 2)
            res['query_density'] = density_map
        if require_contain('query_weights'):
            weight_set = torch.cat(out_weights, dim=0)
            n_sample = weight_set.shape[1]
            weight_all = torch.zeros([total_ray, n_sample], dtype=weight_set.dtype).to(weight_set.device)
            weight_all.scatter_(0, idx.reshape(-1, 1).repeat(1, n_sample), weight_set)
            weight_map = weight_all.reshape(img_wid, img_hei, n_sample).permute(1, 0, 2)
            res['query_weights'] = weight_map

        return res

    def callback_img_visual(self, data, net_out, tag, step):
        """
            The callback function for visualization.
            Notice that the loss will be automatically report and record to writer.
            For image visualization, some custom code is needed.
            Please write image to loss_writer and call flush().
        """
        res = self.visualize_output(resolution_level=4, require_item=['reflectance', 'wrp_viz', 'depth_viz', 'mesh'])
        self.loss_writer.add_image(f'{tag}/img_render', res['wrp_viz'], step)
        self.loss_writer.add_image(f'{tag}/depth_map', res['depth_viz'], step)
        self.loss_writer.add_image(f'{tag}/reflectance-ref', res['reflectance'][0], step)
        self.loss_writer.add_image(f'{tag}/reflectance-rad', res['reflectance'][1], step)
        vertices, triangles = res['mesh']
        self.loss_writer.add_mesh(f'{tag}/mesh', vertices=vertices, faces=triangles, global_step=step)
        self.loss_writer.flush()
