# -*- coding: utf-8 -*-

# @Time:      2023/2/19 18:20
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      exp_classic.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
from configparser import ConfigParser
from pathlib import Path
import numpy as np

from tools.fpp_toolkit import BFHPMPCoder, BFNPMPCoder, GCCPMPCoder, GCOnlyCoder, coord2depth
import pointerlib as plb


# - Coding Part - #
class ExpClassicWorker:
    def __init__(self, args):

        self.args = args

        def safe_path(input_str):
            return None if input_str is '' else Path(input_str)

        self.train_dir = safe_path(self.args.train_dir)
        self.test_dir = safe_path(self.args.test_dir)
        self.res_dir = Path(self.args.out_dir) / self.args.res_dir_name
        self.res_dir.mkdir(exist_ok=True, parents=True)

        self.img_size = None
        self.pat_size = None
        self.coder = None

    def init_all(self):
        print(f'{self.args.argset} output to {self.res_dir} ...', flush=True, end='')
        pass

    def do(self):
        # Dataset
        config = ConfigParser()
        config.read(str(self.train_dir / 'config.ini'), encoding='utf-8')
        self.img_size = plb.str2tuple(config['RawCalib']['img_size'], int)
        self.img_size = self.img_size[1], self.img_size[0]
        self.pat_size = plb.str2tuple(config['RawCalib']['pat_size'], int)
        self.pat_size = self.pat_size[1], self.pat_size[0]
        hei, wid = self.pat_size

        def tag2img(*tags):
            return list(map(lambda tag: self.train_dir / f'img/img_{tag}.png', tags))

        # Coder
        kwargs = {}
        if self.args.argset == 'ClassicBFH':
            wave_length = 80
            self.coder = BFHPMPCoder(hei, wid, wave_length)
            kwargs['pat_low'] = tag2img('pm1280n3i0', 'pm1280n3i1', 'pm1280n3i2')
            kwargs['pat_high'] = tag2img(*[f'pm{wave_length}n3i{i}' for i in (0, 1, 2)])

        elif self.args.argset == 'ClassicBFN':
            self.coder = BFNPMPCoder(hei, wid, 48, 70)
            kwargs['pat_main'] = tag2img('pm48n3i0', 'pm48n3i1', 'pm48n3i2')
            kwargs['pat_sub'] = tag2img('pm70n3i0', 'pm70n3i1', 'pm70n3i2')

        elif self.args.argset == 'ClassicGCC':
            gc_digit = 4
            self.coder = GCCPMPCoder(hei, wid, gc_digit)
            wave_length = int(self.coder.pmp_coder.wave_length)
            kwargs['gray_pats'] = tag2img(*[f'gc{i}' for i in range(gc_digit)])
            kwargs['phase_pats'] = tag2img(*[f'pm{wave_length}n3i{i}' for i in (0, 1, 2)])

        elif self.args.argset == 'ClassicGrayOnly':
            self.coder = GCOnlyCoder(hei, wid, self.args.gc_digit, interpolation=self.args.interpolation)
            kwargs['gray_pats'] = tag2img(*[f'gc{i}' for i in range(self.args.gc_digit)])
            kwargs['gray_pats_inv'] = tag2img(*[f'gc{i}inv' for i in range(self.args.gc_digit)])
            kwargs['mask'] = [self.train_dir / 'gt' / 'mask_occ.png']
            # kwargs['img_base'] = tag2img('uni200', 'uni100')

        coord_wid = self.coder.decode(**kwargs)

        # Save
        save_folder = self.res_dir / f'output/e_00000'
        save_folder.mkdir(parents=True, exist_ok=True)
        plb.imsave(save_folder / 'coord_x.png', coord_wid, scale=50.0, img_type=np.uint16)
        calib_para = [
            plb.str2array(config['RawCalib']['img_intrin'], np.float32),
            plb.str2array(config['RawCalib']['pat_intrin'], np.float32),
            plb.str2array(config['RawCalib']['ext_rot'], np.float32, [3, 3]),
            plb.str2array(config['RawCalib']['ext_tran'], np.float32)
        ]
        depth_map = coord2depth(coord_wid, *calib_para)
        plb.imsave(save_folder / 'depth.png', depth_map, scale=10.0, img_type=np.uint16)

        print('finished.')
        pass
