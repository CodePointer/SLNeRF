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

        pat_dataset = MultiPatDataset(
            scene_folder=self.train_dir,
            pat_idx_set=[0, 1, 2, 3, 4, 5, 6, 7],
            sample_num=self.sample_num,
            calib_para=config['Calibration'],
            device=self.device
        )
        self.train_dataset = None
        self.test_dataset = []
        self.res_writers = []

        if self.args.exp_type == 'train':
            self.save_flag = False
            self.train_dataset = pat_dataset
        elif self.args.exp_type == 'eval':  # Without BP: TIDE
            self.save_flag = self.args.save_res
            self.test_dataset.append(pat_dataset)
            self.res_writers.append(self.create_res_writers())
        else:
            raise NotImplementedError(f'Wrong exp_type: {self.args.exp_type}')

        # Init warp_layer
        self.bound, self.center_pt, self.scale = self.train_dataset.get_bound()
        self.warp_layer = WarpFromXyz(
            calib_para=config['Calibration'],
            pat_mat=self.train_dataset.pat_set,
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
        self.networks['SDFNetwork'] = SDFNetwork(
            d_out=257,
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
        self.networks['SingleVariance'] = SingleVarianceNetwork(
            init_val=0.3
        )
        self.renderer = NeuSLRenderer(
            sdf_network=self.networks['SDFNetwork'],
            deviation_network=self.networks['SingleVariance'],
            color_network=self.warp_layer,
            n_samples=64,
            n_importance=64,
            up_sample_steps=4,
            perturb=1.0
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
        # TODO

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

        # image
        img_input = self.train_dataset.img_set
        rays_o, rays_d, img_size = self.train_dataset.get_uniform_ray(resolution_level=32)
        render_out = self.renderer.render(rays_o, rays_d, self.bound, cos_anneal_ratio=self.anneal_ratio)
        color_fine = render_out['color_fine']  # [N, C]

        img_hei, img_wid = img_size
        img_render_list = []
        for c in range(color_fine.shape[1]):
            img_fine = color_fine[:, c].reshape(img_hei, img_wid)
            img_viz = pvf.img_visual(img_fine)
            img_render_list.append(img_viz)

        # for c in range(color_fine.shape[1]):
        #     img_viz = pvf.img_visual(img_input[c])
        #     img_render_list.append(img_viz)

        wrp_viz = pvf.img_concat(img_render_list, 1, color_fine.shape[1], transpose=True)
        self.loss_writer.add_image(f'{tag}/img_render', wrp_viz, step)

        # Mesh
        vertices, triangles = self.renderer.extract_geometry(
            self.bound[0],
            self.bound[1],
            resolution=64,
            threshold=0.0
        )
        vertices = torch.from_numpy(vertices.astype(np.float32)).unsqueeze(0)
        triangles = torch.from_numpy(triangles.astype(np.int32)).unsqueeze(0)
        self.loss_writer.add_mesh(f'{tag}/mesh', vertices=vertices, faces=triangles, global_step=step)

        self.loss_writer.flush()
