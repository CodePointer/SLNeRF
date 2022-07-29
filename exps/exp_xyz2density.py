# -*- coding: utf-8 -*-
# @Description:
#   Empty worker for new workers writing. This is an example.

# - Package Imports - #
import torch
import torch.nn.functional as F
import numpy as np
from configparser import ConfigParser

from worker.worker import Worker
import pointerlib as plb
from dataset.multi_pat_dataset import MultiPatDataset

from loss.loss_func import SuperviseDistLoss
from networks.layers import WarpFromXyz
from networks.neus import NeuSLRenderer, DensityNetwork


# - Coding Part - #
class ExpXyz2DensityWorker(Worker):
    def __init__(self, args):
        """
            Please add all parameters will be used in the init function.
        """
        super().__init__(args)

        # init_dataset()
        self.pat_dataset = None
        self.sample_num = 512
        self.bound = None
        self.center_pt = None
        self.scale = None
        self.warp_layer = None

        # init_networks()
        self.renderer = None

        # init_losses()
        self.super_loss = None
        self.igr_weight = 0.1

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

        self.pat_dataset = MultiPatDataset(
            scene_folder=self.train_dir,
            # pat_idx_set=[0, 1, 2, 3, 4, 5, 6, 7],
            pat_idx_set=[11],
            sample_num=self.sample_num,
            calib_para=config['Calibration'],
            device=self.device
        )
        self.train_dataset = None
        self.test_dataset = []
        self.res_writers = []

        if self.args.exp_type == 'train':
            self.save_flag = False
            self.train_dataset = self.pat_dataset
        elif self.args.exp_type == 'eval':  # Without BP: TIDE
            self.save_flag = False
            self.test_dataset.append(self.pat_dataset)
            self.res_writers.append(self.create_res_writers())
        else:
            raise NotImplementedError(f'Wrong exp_type: {self.args.exp_type}')

        # Init warp_layer
        self.bound, self.center_pt, self.scale = self.pat_dataset.get_bound()
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
            d_out=1,
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
        self.renderer = NeuSLRenderer(
            sdf_network=self.networks['DensityNetwork'],
            deviation_network=None,
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
        render_out = self.renderer.render_density(data['rays_v'])
        return render_out['color']

    def loss_forward(self, net_out, data):
        """
            How loss functions process the output from network and input data.
            The output will be used with err.backward().
        """
        color_fine = net_out
        total_loss = torch.zeros(1).to(self.device)
        total_loss += self.loss_record(
            'color_l1', pred=color_fine, target=data['color']
        )
        self.avg_meters['Total'].update(total_loss, self.N)
        return total_loss

    def callback_after_train(self, epoch):
        # save_flag
        if self.args.save_stone > 0 and (epoch + 1) % self.args.save_stone == 0:
            self.save_flag = True
        else:
            self.save_flag = False

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

        self.loss_writer.flush()

    def callback_img_visual(self, data, net_out, tag, step):
        """
            The callback function for visualization.
            Notice that the loss will be automatically report and record to writer.
            For image visualization, some custom code is needed.
            Please write image to loss_writer and call flush().
        """
        pvf = plb.VisualFactory
        resolution_level = 4

        # TODO: 看一些PhaseShifting的算法。更换Pattern来看结果是否可行。
        #       然后采集一些数据；包括全黑、全白、GrayCode、PhaseShifting，不同的PhaseShifting模式。
        #       将这些都保存在Data当中，然后进行相应的计算。

        # image
        rays_o, rays_d, img_size = self.pat_dataset.get_uniform_ray(resolution_level=resolution_level)
        rays_o_set = rays_o.split(self.sample_num)
        rays_d_set = rays_d.split(self.sample_num)
        out_rgb_fine = []
        out_depth = []
        # out_pts = []
        # out_min_pts = []
        for rays_o, rays_d in zip(rays_o_set, rays_d_set):
            render_out = self.renderer.render_density(rays_d)
            color_fine = render_out['color']  # [N, C]
            out_rgb_fine.append(color_fine.detach().cpu())

            weights = render_out['weights']
            mid_z_vals = render_out['z_vals']
            max_idx = torch.argmax(weights, dim=1)  # [N]
            mid_z = mid_z_vals[torch.arange(max_idx.shape[0]), max_idx]
            out_depth.append(mid_z.detach().cpu())

            # pts = render_out['pts'].detach().cpu()
            # out_pts.append(pts.reshape(-1, 3))
            # min_pts = pts[torch.arange(max_idx.shape[0]), max_idx]
            # out_min_pts.append(min_pts)

        img_hei, img_wid = img_size
        img_set = torch.cat(out_rgb_fine, dim=0).permute(1, 0)
        channel = img_set.shape[0]
        img_render_list = []
        for c in range(channel):
            img_fine = img_set[c].reshape(img_wid, img_hei).permute(1, 0)
            img_viz = pvf.img_visual(img_fine)
            img_render_list.append(img_viz)
        # plb.imviz(img_render_list[-1], 'img', 10)

        # pts_set = torch.cat(out_pts, dim=0).numpy()
        # np.savetxt(str(self.res_dir / 'depth_query.asc'), pts_set,
        #            fmt='%.2f', delimiter=', ', newline='\n',
        #            encoding='utf-8')
        # min_pts_set = torch.cat(out_min_pts, dim=0).numpy()
        # np.savetxt(str(self.res_dir / 'depth_map.asc'), min_pts_set,
        #            fmt='%.2f', delimiter=', ', newline='\n',
        #            encoding='utf-8')

        img_input = self.pat_dataset.img_set.unsqueeze(0).detach().cpu()
        img_input_re = torch.nn.functional.interpolate(img_input, scale_factor=1 / resolution_level).squeeze(dim=0)
        for c in range(channel):
            img_gt = img_input_re[c]
            img_viz = pvf.img_visual(img_gt)
            img_render_list.append(img_viz)
        # plb.imviz(img_render_list[-1], 'img_gt', 10)
        wrp_viz = pvf.img_concat(img_render_list, 2, channel, transpose=False)
        self.loss_writer.add_image(f'{tag}/img_render', wrp_viz, step)

        depth_set = torch.cat(out_depth, dim=0)
        depth_mat = depth_set.reshape(img_wid, img_hei).permute(1, 0)
        depth_viz = pvf.disp_visual(depth_mat, range_val=self.bound[:, 2].to(depth_mat.device))
        self.loss_writer.add_image(f'{tag}/depth_map', depth_viz, step)

        # plb.imviz(depth_mat, 'depth', 0, normalize=[300, 900])

        # Mesh
        # mesh_bound = np.stack([np.min(pts_set, axis=0), np.max(pts_set, axis=0)])
        # vertices, triangles, pts_set_sdf, pts_select = self.renderer.extract_geometry(
        #     torch.from_numpy(mesh_bound).cuda(),
        #     resolution=128,
        #     threshold=0.0
        # )
        # rays_o, rays_d, img_size = self.pat_dataset.get_uniform_ray(resolution_level=resolution_level)
        # vertices, triangles, pts_set_sdf, pts_select = self.renderer.extract_proj_geometry(
        #     rays_d,
        #     img_size,
        #     self.bound,
        #     z_resolution=128,
        #     threshold=0.0
        # )
        # vertices = torch.from_numpy(vertices.astype(np.float32)).unsqueeze(0)
        # triangles = torch.from_numpy(triangles.astype(np.int32)).unsqueeze(0)
        # ver_color = self.warp_layer(vertices[0].cuda()).detach().cpu()
        # vertex_color = (ver_color[:, :1] * 255).numpy().astype(np.uint8).repeat(3, axis=1)
        # # face_color = np.zeros()
        # mesh = trimesh.Trimesh(vertices[0], triangles[0], vertex_colors=vertex_color)
        # mesh.export(str(self.res_dir / 'mesh.ply'))
        # self.loss_writer.add_mesh(f'{tag}/mesh', vertices=vertices, faces=triangles, global_step=step)
        # np.savetxt(str(self.res_dir / 'sdf_query.asc'), pts_set_sdf.numpy(),
        #            fmt='%.2f', delimiter=', ', newline='\n', encoding='utf-8')
        # np.savetxt(str(self.res_dir / 'sdf_select.asc'), pts_select.numpy(),
        #            fmt='%.2f', delimiter=', ', newline='\n', encoding='utf-8')
        # print('sdf_query finished.')

        self.loss_writer.flush()
