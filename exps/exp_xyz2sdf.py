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
                'img_list', 'wrp_viz', 'depth_viz', 'depth_map', 'point_cloud'
            ])  # TODO: 可视化所有的query部分，用于绘制结果图。
            save_folder = self.res_dir / 'output' / f'e_{epoch:05}'
            save_folder.mkdir(parents=True, exist_ok=True)
            plb.imsave(save_folder / f'wrp_viz.png', res['wrp_viz'])
            plb.imsave(save_folder / f'{self.args.run_tag}_viz.png', res['depth_viz'])
            plb.imsave(save_folder / f'{self.args.run_tag}.png', res['depth_map'], scale=10.0, img_type=np.uint16)
            np.savetxt(str(save_folder / f'{self.args.run_tag}.asc'), res['point_cloud'],
                       fmt='%.2f', delimiter=',', newline='\n', encoding='utf-8')
            img_folder = save_folder / 'img'
            for i, img in enumerate(res['img_list']):
                plb.imsave(img_folder / f'img_{i}.png', img, mkdir=True)

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

        # TODO: Write out the visualization part for SDF version.
        #       Please refer to https://github.com/Totoro97/NeuS/blob/main/exp_runner.py.
        #       validate_image & validate_sdf would help.
        #       Maybe just extract the sdf is enough even we haven't finish the training.
        res = {}
        if require_contain('wrp_viz'):
            # image
            rays_o, rays_d, img_size, mask_val, _ = self.pat_dataset.get_uniform_ray(resolution_level=resolution_level)
            rays_o_set = rays_o.split(self.sample_num)  # TODO: Maybe we have to change this uniform ray sampling.
            rays_d_set = rays_d.split(self.sample_num)

            out_rgb_fine = []
            for rays_o, rays_d in tqdm(zip(rays_o_set, rays_d_set)):
                render_out = self.renderer.render(rays_o, rays_d, self.bound,
                                                  cos_anneal_ratio=self.anneal_ratio)
                color_fine = render_out['color_fine']  # [N, C]
                out_rgb_fine.append(color_fine.detach().cpu())
                del render_out

            img_hei, img_wid = img_size
            img_set = torch.cat(out_rgb_fine, dim=0).permute(1, 0)
            channel = img_set.shape[0]
            img_render_list = []
            for c in range(channel):
                img_fine = img_set[c].reshape(img_wid, img_hei).permute(1, 0)
                img_viz = pvf.img_visual(img_fine)
                img_render_list.append(img_viz)
            img_input = self.pat_dataset.img_set.unsqueeze(0).detach().cpu()
            img_input_re = torch.nn.functional.interpolate(img_input, scale_factor=1 / resolution_level).squeeze()
            for c in range(channel):
                img_gt = img_input_re[c]
                img_viz = pvf.img_visual(img_gt)
                img_render_list.append(img_viz)
            res['wrp_viz'] = pvf.img_concat(img_render_list, 2, channel, transpose=False)

        if require_contain('mesh'):
            # bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
            # bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)
            #
            # vertices, triangles = \
            #     self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
            # os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
            #
            # mesh = trimesh.Trimesh(vertices, triangles)
            # mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))
            # TODO. 这一部分是从代码里面直接粘过来的。存在各种问题。
            pass


        return res

    def callback_img_visual(self, data, net_out, tag, step):
        """
            The callback function for visualization.
            Notice that the loss will be automatically report and record to writer.
            For image visualization, some custom code is needed.
            Please write image to loss_writer and call flush().
        """
        pvf = plb.VisualFactory
        resolution_level = 4

        # image
        rays_o, rays_d, img_size = self.pat_dataset.get_uniform_ray(resolution_level=resolution_level)
        rays_o_set = rays_o.split(self.sample_num)
        rays_d_set = rays_d.split(self.sample_num)
        out_rgb_fine = []
        out_depth = []
        out_pts = []
        out_min_pts = []
        for rays_o, rays_d in tqdm(zip(rays_o_set, rays_d_set)):
            render_out = self.renderer.render(rays_o, rays_d, self.bound,
                                              cos_anneal_ratio=self.anneal_ratio)
            color_fine = render_out['color_fine']  # [N, C]
            out_rgb_fine.append(color_fine.detach().cpu())
            weights = render_out['weights']
            mid_z_vals = render_out['mid_z_vals']
            max_idx = torch.argmax(weights, dim=1)  # [N]
            mid_z = mid_z_vals[torch.arange(max_idx.shape[0]), max_idx]
            out_depth.append(mid_z.detach().cpu())
            pts = render_out['pts'].detach().cpu()
            out_pts.append(pts.reshape(-1, 3))
            min_pts = pts[torch.arange(max_idx.shape[0]), max_idx]
            out_min_pts.append(min_pts)
        img_hei, img_wid = img_size
        img_set = torch.cat(out_rgb_fine, dim=0).permute(1, 0)
        channel = img_set.shape[0]
        img_render_list = []
        for c in range(channel):
            img_fine = img_set[c].reshape(img_wid, img_hei).permute(1, 0)
            img_viz = pvf.img_visual(img_fine)
            img_render_list.append(img_viz)
        depth_set = torch.cat(out_depth, dim=0)
        depth_mat = depth_set.reshape(img_wid, img_hei).permute(1, 0)
        plb.imviz(img_render_list[-1], 'img', 10)
        plb.imviz(depth_mat, 'depth', 10, normalize=[300, 900])

        pts_set = torch.cat(out_pts, dim=0).numpy()
        np.savetxt(str(self.res_dir / 'depth_query.asc'), pts_set,
                   fmt='%.2f', delimiter=', ', newline='\n',
                   encoding='utf-8')
        min_pts_set = torch.cat(out_min_pts, dim=0).numpy()
        np.savetxt(str(self.res_dir / 'depth_map.asc'), min_pts_set,
                   fmt='%.2f', delimiter=', ', newline='\n',
                   encoding='utf-8')
        # with open(str(self.res_dir / 'depth_query.asc'), 'w+', encoding='utf-8') as file:
        #     for pts in pts_set:
        #         file.write(f'{pts[0]:.02f}, {pts[1]:.02f}, {pts[2]:.02f}\n')
        print('depth_query finished.')

        # channel = 8
        # img_hei, img_wid = 240, 320
        img_input = self.pat_dataset.img_set.unsqueeze(0).detach().cpu()
        img_input_re = torch.nn.functional.interpolate(img_input, scale_factor=1 / resolution_level).squeeze()
        for c in range(channel):
            img_gt = img_input_re[c]
            img_viz = pvf.img_visual(img_gt)
            img_render_list.append(img_viz)
        plb.imviz(img_render_list[-1], 'img_gt', 10)

        wrp_viz = pvf.img_concat(img_render_list, 2, channel, transpose=False)
        self.loss_writer.add_image(f'{tag}/img_render', wrp_viz, step)

        # Mesh
        # mesh_bound = np.stack([np.min(pts_set, axis=0), np.max(pts_set, axis=0)])
        # vertices, triangles, pts_set_sdf, pts_select = self.renderer.extract_geometry(
        #     torch.from_numpy(mesh_bound).cuda(),
        #     resolution=128,
        #     threshold=0.0
        # )
        rays_o, rays_d, img_size = self.pat_dataset.get_uniform_ray(resolution_level=resolution_level)
        vertices, triangles, pts_set_sdf, pts_select = self.renderer.extract_proj_geometry(
            rays_d,
            img_size,
            self.bound,
            z_resolution=128,
            threshold=0.0
        )

        vertices = torch.from_numpy(vertices.astype(np.float32)).unsqueeze(0)
        triangles = torch.from_numpy(triangles.astype(np.int32)).unsqueeze(0)
        ver_color = self.warp_layer(vertices[0].cuda()).detach().cpu()
        vertex_color = (ver_color[:, :1] * 255).numpy().astype(np.uint8).repeat(3, axis=1)
        # face_color = np.zeros()
        mesh = trimesh.Trimesh(vertices[0], triangles[0], vertex_colors=vertex_color)
        mesh.export(str(self.res_dir / 'mesh.ply'))
        self.loss_writer.add_mesh(f'{tag}/mesh', vertices=vertices, faces=triangles, global_step=step)

        np.savetxt(str(self.res_dir / 'sdf_query.asc'), pts_set_sdf.numpy(),
                   fmt='%.2f', delimiter=', ', newline='\n', encoding='utf-8')
        np.savetxt(str(self.res_dir / 'sdf_select.asc'), pts_select.numpy(),
                   fmt='%.2f', delimiter=', ', newline='\n', encoding='utf-8')
        print('sdf_query finished.')

        self.loss_writer.flush()
