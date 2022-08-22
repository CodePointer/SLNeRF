# -*- coding: utf-8 -*-

# @Time:      2022/8/16 17:01
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      sensors.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
import numpy as np
from configparser import ConfigParser
from pointerlib import plb


# - Coding Part - #
class BaseCamera:
    def __init__(self):
        self.img_size = None

    def set_parameters(self):
        pass

    def get_resolution(self):
        return self.img_size

    def capture(self, **kwargs):
        raise NotImplementedError('Not implemented by BaseCamera')


class VirtualCamera(BaseCamera):
    def __init__(self, calib_ini, depth_cam):
        """
        Render img virtually.
        :param calib_ini: <pathlib.Path>. Calibrated function included.
        :param depth_cam: <numpy.Array> or <torch.Tensor>. Shape: [H, W, 1] or [1, H, W]
        """
        super(VirtualCamera).__init__()

        config = ConfigParser()
        config.read(str(calib_ini), encoding='utf-8')
        coord_func = plb.CoordCompute(
            cam_size=plb.str2tuple(config['Calibration']['img_size'], int),
            cam_intrin=plb.str2tuple(config['Calibration']['img_intrin'], np.float32),
            pat_intrin=plb.str2tuple(config['Calibration']['pat_intrin'], np.float32),
            ext_rot=plb.str2array(config['Calibration']['ext_rot'], np.float32, [3, 3]),
            ext_tran=plb.str2array(config['Calibration']['ext_tran'], np.float32)
        )
        self.uu, self.vv = coord_func(depth_cam)
        self.warp_func = plb.WarpLayer2D()

    def render(self, img_pat, color):
        if color is None:
            return img_pat
        base_light = 0.2
        back_light = 0.3
        img_base = color * (1 - base_light) + base_light
        img_render = img_base * ((1 - back_light) * img_pat + back_light)
        return img_render

    def capture(self, pat, color=None, mask=None):
        img_pat = self.warp_func(self.uu, self.vv, pat, mask_flag=True)
        if mask is not None:
            img_pat *= mask
        return self.render(img_pat, color)


class BaseProjector:
    def __init__(self):
        self.pat_size = None
        self.patterns = []
        pass

    def set_pattern_set(self, pattern_set):
        self.pat_size = pattern_set[0].shape
        self.patterns.append(*pattern_set)
        return len(self.patterns)

    def project(self, idx):
        return self.patterns[idx]
