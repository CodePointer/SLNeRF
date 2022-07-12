# -*- coding: utf-8 -*-
# @Description:
#   Empty worker for new workers writing. This is an example.

# - Package Imports - #
import torch
import numpy as np
from configparser import ConfigParser

from worker.worker import Worker
import pointerlib as plb
from dataset.multi_pat_dataset import MultiPatDataset
from networks.bacon import MultiscaleBACON
from loss.loss_func import SuperviseDistLoss
from networks.layers import WarpFromDepth


# - Coding Part - #
class ExpXy2DepthWorker(Worker):
    def __init__(self, args):
        """
            Please add all parameters will be used in the init function.
        """
        super().__init__(args)

        # init_dataset()
        self.warp_layer = None

        # init_networks()
        self.output_layer_idx = [1, 2, 4]
        self.multi_scale = len(self.output_layer_idx)

        # init_losses()
        self.super_loss = None

    def init_dataset(self):
        """
            Requires:
                self.train_dataset
                self.test_dataset
                self.res_writers  (self.create_res_writers)
        """
        pat_dataset = MultiPatDataset(self.train_dir, pat_idx_set=[0, 1, 2, 3, 4, 5, 6, 7],
                                      device=self.device)
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
        config = ConfigParser()
        config.read(str(self.train_dir / 'config.ini'), encoding='utf-8')
        self.warp_layer = WarpFromDepth(config['Calibration'], device=self.device)

        self.logging(f'--train_dir: {self.train_dir}')

    def init_networks(self):
        """
            Requires:
                self.networks (dict.)
            Keys will be used for network saving.
        """
        self.networks['MultiBacon'] = MultiscaleBACON(
            in_size=2,
            hidden_size=256,
            out_size=1,
            hidden_layers=4,
            weight_scale=1.0,
            bias=True,
            output_act=False,
            frequency=(256, 256),
            quantization_interval=2 * np.pi,
            centered=True,
            is_sdf=False,
            input_scales=[1/8, 1/8, 1/4, 1/4, 1/4],
            output_layers=[1, 2, 4],
            reuse_filters=False
        )
        self.logging(f'--networks: {",".join(self.networks.keys())}')
        self.logging(f'--networks-static: {",".join(self.network_static_list)}')

    def init_losses(self):
        """
            Requires:
                self.loss_funcs (dict.)
            Keys will be used for avg_meter.
        """
        self.super_loss = SuperviseDistLoss(dist='l2')
        for s in range(self.multi_scale):
            self.loss_funcs[f'ph{s}'] = self.super_loss
            self.loss_funcs[f'direct{s}'] = self.super_loss
        self.logging(f'--loss types: {self.loss_funcs.keys()}')
        pass

    def data_process(self, idx, data):
        """
            Process data and move data to self.device.
            output will be passed to :net_forward().
        """
        return data

    def net_forward(self, data):
        """
            How networks process input data and give out network output.
            The output will be passed to :loss_forward().
        """
        depth_set = self.networks['MultiBacon'](data['coord'])

        depth_out = []
        hei_img, wid_img = data['img'][0].shape[-2:]
        for depth_1d in depth_set:
            depth_map = depth_1d.permute(0, 2, 1).reshape(self.N, 1, hei_img, wid_img)
            # 放缩：[-8,8] -> [0, 1000]
            depth_map = (depth_map + 8.0) / 16.0 * 1e3
            depth_out.append(depth_map)

        return depth_out  # [depth_1d] * multi_scale

    def loss_forward(self, net_out, data):
        """
            How loss functions process the output from network and input data.
            The output will be used with err.backward().
        """
        depth_out = net_out
        total_loss = torch.zeros(1).to(self.device)

        # Warp by depth.
        img_wrp_set = []
        for depth_map in depth_out:
            img_wrp = self.warp_layer(depth_mat=depth_map, src_mat=data['pat'])  # [N, C, Hi, Wi]
            img_wrp_set.append(img_wrp)

        # Compare img & pat
        for s in range(self.multi_scale):
            total_loss += self.loss_record(
                f'ph{s}', pred=img_wrp_set[s], target=data['img']
            ) * 1.0

        # Debug: direct with depth map
        for s in range(self.multi_scale):
            total_loss += self.loss_record(
                f'direct{s}', pred=depth_out[s], target=data['depth']
            ) * 0.0

        self.avg_meters['Total'].update(total_loss, self.N)
        return total_loss

    def callback_after_train(self, epoch):
        # save_flag
        if self.args.save_stone > 0 and (epoch + 1) % self.args.save_stone == 0:
            self.save_flag = True
        else:
            self.save_flag = False

    def callback_save_res(self, data, net_out, dataset, res_writer):
        """
            The callback function for data saving.
            The data should be saved with the input.
            Please create new folders and save result.
        """
        out_dir, config = res_writer
        for i in range(len(net_out)):
            plb.imsave(out_dir / f'depth_s{i}.png', net_out[i], scale=10.0, img_type=np.uint16)

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
        multi_scale = len(net_out)
        pvf = plb.VisualFactory

        depth_list = []
        img_wrp_list = []
        for s in range(multi_scale):
            depth_map = net_out[s]
            # TODO: Visualize depth
            depth_map_viz = pvf.disp_visual(depth_map, range_val=[0.0, 1e3])
            depth_list.append(depth_map_viz)
            img_wrp = self.warp_layer(depth_mat=depth_map, src_mat=data['pat'])  # [N, C, Hi, Wi]
            for c in range(img_wrp.shape[1]):
                img_wrp_viz = pvf.img_visual(img_wrp[0, c])
                img_wrp_list.append(img_wrp_viz)

        depth_gt_viz = pvf.disp_visual(data['depth'], range_val=[0.0, 1e3])
        depth_list.append(depth_gt_viz)

        depth_viz = pvf.img_concat(depth_list, 1, multi_scale + 1)
        self.loss_writer.add_image(f'{tag}/depth_est', depth_viz, step)

        for c in range(data['img'].shape[1]):
            img_viz = pvf.img_visual(data['img'][0, c])
            img_wrp_list.append(img_viz)

        wrp_viz = pvf.img_concat(img_wrp_list, multi_scale + 1, data['img'].shape[1], transpose=True)
        self.loss_writer.add_image(f'{tag}/img_wrp', wrp_viz, step)

        self.loss_writer.flush()
