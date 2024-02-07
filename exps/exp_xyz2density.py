# -*- coding: utf-8 -*-
# @Description:
#   Empty worker for new workers writing. This is an example.

# - Package Imports - #
import torch
import torch.nn.functional as F
import numpy as np
from configparser import ConfigParser
import matplotlib.pyplot as plt
from tqdm import tqdm

from worker.worker import Worker
import pointerlib as plb
from dataset.dataset_neus import MultiPatDataset, MultiPatDatasetUni

from loss.loss_func import SuperviseDistLoss
from networks.layers import WarpFromXyz, WarpFromXyzUni, apply_4x4mat
from networks.neural_density import NeuSLRenderer, DensityNetwork, ReflectNetwork


# - Coding Part - #
class ExpXyz2DensityWorker(Worker):
    def __init__(self, args):
        """
            Please add all parameters will be used in the init function.
        """
        super().__init__(args)

        # Fix for NeRF-like training.
        self.N = 1
        self.sample_num = self.args.batch_num
        self.train_shuffle = False

        # init_dataset()
        self.focus_len = None
        self.pat_dataset = None
        self.warp_layer = None

        # init_networks()
        self.renderer = None

        self.parameters = {
            'render': plb.ParamScheduler(self.args.render_scheduler, float),
            'pat_num': plb.ParamScheduler(self.args.patnum_scheduler, int),
            'warp': plb.ParamScheduler(self.args.warp_scheduler, float),
        }

    def init_dataset(self):
        """
            Requires:
                self.train_dataset
                self.test_dataset
                self.res_writers  (self.create_res_writers)
        """
        config = ConfigParser()
        config.read(str(self.train_dir / 'config.ini'), encoding='utf-8')

        pat_idx_set = [x.strip() for x in self.args.pat_set.split(',')]
        ref_img_set = [x.strip() for x in self.args.reflect_set.split(',')]
        self.focus_len = plb.str2array(config['RawCalib']['img_intrin'], np.float32)[0]
        self.pat_dataset = MultiPatDatasetUni(
            scene_folder=self.train_dir,
            pat_idx_set=pat_idx_set,
            ref_img_set=ref_img_set,
            sample_num=self.sample_num,
            calib_para=config['RawCalib'],
            device=self.device
        )
        self.train_dataset = self.pat_dataset

        # Set pattern number
        pat_num = self.parameters['pat_num'].get_value()
        if pat_num < 0:
            self.parameters['pat_num'].value = len(pat_idx_set)

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
        self.networks['DensityNetwork'] = DensityNetwork(
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
        config = ConfigParser()
        config.read(str(self.train_dir / 'config.ini'), encoding='utf-8')
        self.warp_layer = WarpFromXyzUni(
            calib_para=config['RawCalib'],
            pat_mat=self.pat_dataset.get_pat_set(),
            w2c=self.pat_dataset.get_w2c(),
            device=self.device
        )
        self.renderer = NeuSLRenderer(
            sdf_network=self.networks['DensityNetwork'],
            deviation_network=None,
            color_network=self.warp_layer,
            n_samples=32,
            n_importance=32,
            up_sample_steps=4,
            perturb=1.0
        )
        self.logging(f'Networks: {",".join(self.networks.keys())}')
        self.logging(f'Networks-static: {",".join(self.network_static_list)}')

    def init_losses(self):
        """
            Requires:
                self.loss_funcs (dict.)
            Keys will be used for avg_meter.
        """
        self.loss_funcs['render_color'] = SuperviseDistLoss(dist='l1')
        self.loss_funcs['surface_color'] = SuperviseDistLoss(dist='l1')
        self.logging(f'Loss types: {self.loss_funcs.keys()}')
        pass

    def data_process(self, idx, data):
        """
            Process data and move data to self.device.
            output will be passed to :net_forward().
        """
        for key in data:
            data[key] = data[key][0]  # Remove batch number.
        return data

    def prepare_training(self):
        self.n_iter = self.args.iter_start
        self.pat_dataset.iter_times = self.args.total_iter - self.n_iter

        self.epoch_start = 0
        self.epoch_end = 1

        # Load network
        self._net_load(self.n_iter - 1)

    def net_forward(self, data):
        """
            How networks process input data and give out network output.
            The output will be passed to :loss_forward().
        """
        near, far = self.pat_dataset.near_far_from_sphere(data['rays_o'], data['rays_v'])
        render_out = self.renderer.render_density(
            rays_o=data['rays_o'],
            rays_d=data['rays_v'],
            reflect=data['reflect'],
            near=near,
            far=far
        )
        return render_out

    def loss_forward(self, net_out, data):
        """
            How loss functions process the output from network and input data.
            The output will be used with err.backward().
        """
        # color_fine, depth_res, weights = net_out
        total_loss = torch.zeros(1).to(self.device)
        pat_num = self.parameters['pat_num'].get_value()

        # render color loss
        render_color_loss = self.loss_record(
            'render_color', pred=net_out['color'][:, :pat_num],
              target=data['color'][:, :pat_num], mask=data['mask'],
              scale=1.0 / float(pat_num)
        )
        total_loss += render_color_loss * self.parameters['render'].get_value()

        # surface color loss
        surface_color_loss = self.loss_record(
            'surface_color', pred=net_out['color_1pt'][:, :pat_num],
              target=data['color'][:, :pat_num], mask=data['mask'],
              scale=1.0 / float(pat_num)
        )
        total_loss += surface_color_loss * self.parameters['warp'].get_value()

        self.avg_meters['Total'].update(total_loss, self.N)
        return total_loss

    def callback_after_iteration(self, idx, data, epoch):
        # Update parameters
        for key in self.parameters:
            self.parameters[key].update(self.n_iter)

    def callback_realtime_report(self, batch_idx, batch_total, epoch, tag, step, log_flag=True):
        super().callback_realtime_report(batch_idx, batch_total, epoch, tag, step, log_flag)
        for key in self.parameters:
            val = self.parameters[key].get_value()
            if log_flag:
                self.loss_writer.add_scalar(f'{tag}-para/{key}', val, self.n_iter)
        self.loss_writer.flush()

    def callback_img_visual(self, data, net_out, tag, step):
        """
            The callback function for visualization.
            Notice that the loss will be automatically report and record to writer.
            For image visualization, some custom code is needed.
            Please write image to loss_writer and call flush().
        """
        res = self.visualize_output(resolution_level=4, require_item=['wrp_viz', 'depth_viz', 'mesh'])
        self.loss_writer.add_image(f'{tag}/img_render', res['wrp_viz'], step)
        self.loss_writer.add_image(f'{tag}/depth_map', res['depth_viz'], step)
        vertices, triangles = res['mesh']
        vertices[..., 1:] = - vertices[..., 1:]  # Coordinate transport
        self.loss_writer.add_mesh(f'{tag}/mesh', vertices=vertices, faces=triangles, global_step=step)
        self.loss_writer.flush()

    def check_save_res(self, **kwargs):
        """
            Rewrite the report.
        """
        if self.args.debug_mode:
            return True
        if self.args.save_stone == 0:
            return False
        else:
            return self.n_iter % self.args.save_stone == 0 or ((self.n_iter + 100) % self.args.save_stone == 0)

    def callback_save_res(self, epoch, data, net_out, dataset):
        """
            The callback function for data saving.
            The data should be saved with the input.
            Please create new folders and save result.
        """
        # Save
        res = self.visualize_output(resolution_level=1, require_item=[
            'img_list', 'wrp_viz', 'depth_viz', 'depth_map', 'point_cloud'
        ])
        save_folder = self.res_dir / 'output' / f'iter_{epoch:05}'
        save_folder.mkdir(parents=True, exist_ok=True)
        plb.imsave(save_folder / f'wrp_viz.png', res['wrp_viz'])
        plb.imsave(save_folder / f'depth_viz.png', res['depth_viz'])
        plb.imsave(save_folder / f'depth.png', res['depth_map'], scale=10.0, img_type=np.uint16)
        np.savetxt(str(save_folder / f'pcd.asc'), res['point_cloud'],
                   fmt='%.2f', delimiter=',', newline='\n', encoding='utf-8')
        img_folder = save_folder / 'img'
        for i, img in enumerate(res['img_list']):
            plb.imsave(img_folder / f'img_{i}.png', img, mkdir=True)

        # Save network
        self._net_save(self.n_iter)

        pass

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
        rays_o, rays_d, reflect, mask_val = self.pat_dataset.gen_uniform_ray(
            resolution_level=resolution_level
        )
        img_hei, img_wid = rays_o.shape[-2:]

        rays_o = rays_o.reshape(rays_o.shape[0], -1).permute(1, 0)
        rays_d = rays_d.reshape(rays_d.shape[0], -1).permute(1, 0)
        reflect = reflect.reshape(reflect.shape[0], -1).permute(1, 0)
        mask_val = mask_val.reshape(-1)

        # total_ray = rays_o.shape[0]
        # idx = torch.arange(0, total_ray, dtype=torch.long)
        # idx = idx[mask_val > 0.0]
        # rays_o = rays_o[mask_val > 0.0]
        # rays_d = rays_d[mask_val > 0.0]
        # reflect = reflect[mask_val > 0.0]

        rays_o_set = rays_o.split(self.sample_num)
        rays_d_set = rays_d.split(self.sample_num)
        reflect_set = reflect.split(self.sample_num)

        out_rgb_fine = []
        out_rgb_1pt = []
        out_reflectance = []
        out_depth = []
        out_z = []
        out_weights = []
        out_pts = []
        out_color = []
        out_density = []
        pbar = tqdm(total=len(rays_o_set), desc=f'Vis-Resolution={resolution_level}')
        for rays_o, rays_d, reflect in zip(rays_o_set, rays_d_set, reflect_set):
            pbar.update(1)
            near, far = self.pat_dataset.near_far_from_sphere(rays_o, rays_d)
            render_out = self.renderer.render_density(
                rays_o, rays_d, reflect, near, far
            )

            if require_contain('img_list', 'wrp_viz'):
                color_fine = render_out['color']  # [N, C]
                out_rgb_fine.append(color_fine.detach().cpu())
                out_rgb_1pt.append(render_out['color_1pt'].detach().cpu())

            if require_contain('depth_map', 'depth_viz', 'point_cloud', 'mesh'):
                # weights = render_out['weights']
                # mid_z_vals = render_out['pts'][:, :, -1]
                # max_idx = torch.argmax(weights, dim=1)  # [N]
                # mid_z = mid_z_vals[torch.arange(max_idx.shape[0]), max_idx]
                # out_depth.append(mid_z.detach().cpu())
                pts_sum = render_out['pts_sum'].detach().cpu()
                # scale_mat = torch.from_numpy(self.pat_dataset.get_scale_mat())
                # pts_sum_wrd = pts_sum * scale_mat[0, 0] + scale_mat[:3, 3][None]
                w2c = self.pat_dataset.get_w2c()
                pts_sum_wrd = apply_4x4mat(pts_sum, w2c)
                depth_val = pts_sum_wrd[:, -1:]
                out_depth.append(depth_val)

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
        pbar.close()

        res = {}
        if require_contain('img_list', 'wrp_viz'):
            # img_set = torch.cat(out_rgb_fine, dim=0).permute(1, 0)
            # channel = img_set.shape[0]
            # img_all = torch.zeros([channel, total_ray], dtype=img_set.dtype).to(img_set.device)
            # img_all.scatter_(1, idx.reshape(1, -1).repeat(channel, 1), img_set)
            img_render_list = []
            if len(out_rgb_fine) > 0:
                img_fine = torch.cat(out_rgb_fine, dim=0).permute(1, 0)
                img_fine = img_fine.reshape(-1, img_hei, img_wid)
                for c in range(img_fine.shape[0]):
                    img_viz = pvf.img_visual(img_fine[c])
                    img_render_list.append(img_viz)

            # img_set = torch.cat(out_rgb_1pt, dim=0).permute(1, 0)
            # channel = img_set.shape[0]
            # img_all = torch.zeros([channel, total_ray], dtype=img_set.dtype).to(img_set.device)
            # img_all.scatter_(1, idx.reshape(1, -1).repeat(channel, 1), img_set)
            img_warp_list = []
            if len(out_rgb_1pt) > 0:
                img_1pt = torch.cat(out_rgb_1pt, dim=0).permute(1, 0)
                img_1pt = img_1pt.reshape(-1, img_hei, img_wid)
                for c in range(img_1pt.shape[0]):
                    img_viz = pvf.img_visual(img_1pt[c])
                    img_warp_list.append(img_viz)

            if require_contain('img_list'):
                res['img_list'] = img_render_list

            if require_contain('wrp_viz'):
                img_gt_list = []
                img_gt_set = self.pat_dataset.get_cut_img(resolution_level)
                for c in range(img_gt_set.shape[0]):
                    img_viz = pvf.img_visual(img_gt_set[c].detach().cpu())
                    img_gt_list.append(img_viz)
                wrp_viz = pvf.img_concat(img_warp_list + img_render_list + img_gt_list, 3, img_gt_set.shape[0], transpose=False)
                res['wrp_viz'] = wrp_viz

        if require_contain('depth_map', 'depth_viz', 'point_cloud', 'mesh'):
            depth_set = torch.cat(out_depth, dim=0).reshape(-1)
            # depth_all = torch.zeros([total_ray], dtype=depth_set.dtype).to(depth_set.device)
            # depth_all.scatter_(0, idx, depth_set)
            depth_map = depth_set.reshape(img_hei, img_wid)
            if require_contain('depth_map'):
                res['depth_map'] = depth_map
            if require_contain('depth_viz'):
                depth_viz = pvf.disp_visual(depth_map, range_val=self.pat_dataset.z_range)
                res['depth_viz'] = depth_viz
            if require_contain('point_cloud', 'mesh'):
                # TODO: DepthMapVisual + mask
                visualizer = plb.DepthMapConverter(depth_map.unsqueeze(0), focus=self.focus_len / resolution_level)
                if require_contain('point_cloud'):
                    res['point_cloud'] = visualizer.to_xyz_set()
                if require_contain('mesh'):
                    res['mesh'] = visualizer.to_mesh()

        if require_contain('query_z'):
            z_set = torch.cat(out_z, dim=0)  # [N, n_sample]
            n_sample = z_set.shape[1]
            # z_all = torch.zeros([total_ray, n_sample], dtype=z_set.dtype).to(z_set.device)
            # z_all.scatter_(0, idx.reshape(-1, 1).repeat(1, n_sample), z_set)
            z_map = z_set.reshape(img_hei, img_wid, n_sample)
            res['query_z'] = z_map

        if require_contain('query_points'):
            pts_set = torch.cat(out_pts, dim=0)  # [N, n_sample, 3]
            n_sample = pts_set.shape[1]
            # pts_all = torch.zeros([total_ray, n_sample, 3], dtype=pts_set.dtype).to(pts_set.device)
            # pts_all.scatter_(0, idx.reshape(-1, 1, 1).repeat(1, n_sample, 3), pts_set)
            pts_map = pts_set.reshape(img_hei, img_wid, n_sample, 3)
            res['query_points'] = pts_map
        if require_contain('query_density'):
            density_set = torch.cat(out_density, dim=0)  # [N, n_sample]
            n_sample = density_set.shape[1]
            # density_all = torch.zeros([total_ray, n_sample], dtype=density_set.dtype).to(density_set.device)
            # density_all.scatter_(0, idx.reshape(-1, 1).repeat(1, n_sample), density_set)
            density_map = density_set.reshape(img_hei, img_wid, n_sample)
            res['query_density'] = density_map
        if require_contain('query_weights'):
            weight_set = torch.cat(out_weights, dim=0)
            n_sample = weight_set.shape[1]
            # weight_all = torch.zeros([total_ray, n_sample], dtype=weight_set.dtype).to(weight_set.device)
            # weight_all.scatter_(0, idx.reshape(-1, 1).repeat(1, n_sample), weight_set)
            weight_map = weight_set.reshape(img_hei, img_wid, n_sample)
            res['query_weights'] = weight_map

        return res
