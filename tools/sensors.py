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
import cv2
import torch
from configparser import ConfigParser
from pointerlib import plb
import PySpin
import cv2


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
    def __init__(self, calib_ini, depth_cam, virtual_projector):
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
            cam_intrin=plb.str2array(config['Calibration']['img_intrin'], np.float32),
            pat_intrin=plb.str2array(config['Calibration']['pat_intrin'], np.float32),
            ext_rot=plb.str2array(config['Calibration']['ext_rot'], np.float32, [3, 3]),
            ext_tran=plb.str2array(config['Calibration']['ext_tran'], np.float32)
        )
        self.uu, self.vv = coord_func(depth_cam)
        self.warp_func = plb.WarpLayer2D()

        self.virtual_projector = virtual_projector

    def render(self, img_pat, color):
        if color is None:
            return img_pat
        base_light = 0.2
        back_light = 0.3
        img_base = color * (1 - base_light) + base_light
        img_render = img_base * ((1 - back_light) * img_pat + back_light)

        # TODO: Add noise.
        gaussian_noise_scale = 0.02
        g_noise = torch.randn_like(img_render) * gaussian_noise_scale
        img_render = (img_render + g_noise).clamp(0.0, 1.0)

        return img_render[0]

    def capture(self, pat=None, color=None, mask=None):
        if pat is None:
            pat_idx = self.virtual_projector.current_idx
            pat = self.virtual_projector.patterns[pat_idx]
        img_pat = self.warp_func(self.uu, self.vv, pat, mask_flag=True)
        if mask is not None:
            img_pat *= mask
        return self.render(img_pat, color)


class Flea3Camera(BaseCamera):
    def __init__(self, cam_idx=0, avg_frame=1):
        super().__init__()

        self.system = PySpin.System.GetInstance()

        self.cam_list = self.system.GetCameras()
        assert cam_idx < self.cam_list.GetSize()

        self.camera = self.cam_list[cam_idx]
        # self.camera.GetTLDeviceNodeMap()
        self.camera.Init()

        node_map = self.camera.GetNodeMap()
        self.node_map = node_map

        # Mono8
        node_pixel_format = PySpin.CEnumerationPtr(node_map.GetNode('PixelFormat'))
        if PySpin.IsAvailable(node_pixel_format) and PySpin.IsWritable(node_pixel_format):
            node_pixel_format_mono8 = PySpin.CEnumEntryPtr(node_pixel_format.GetEntryByName('Mono8'))
            if PySpin.IsAvailable(node_pixel_format_mono8) and PySpin.IsReadable(node_pixel_format_mono8):
                pixel_format_mono8 = node_pixel_format_mono8.GetValue()
                node_pixel_format.SetIntValue(pixel_format_mono8)

        # Width
        node_width = PySpin.CIntegerPtr(node_map.GetNode('Width'))
        width = node_width.GetMax()
        node_height = PySpin.CIntegerPtr(node_map.GetNode('Height'))
        height = node_height.GetMax()
        self.img_size = (height, width)

        self.avg_frame = avg_frame
        pass

    def __del__(self):
        del self.camera
        self.cam_list.Clear()
        self.system.ReleaseInstance()

    def capture_images(self, frame_num):
        # In order to access the node entries, they have to be casted to a pointer type (CEnumerationPtr here)
        node_acquisition_mode = PySpin.CEnumerationPtr(self.node_map.GetNode('AcquisitionMode'))
        if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
            return False

        # Retrieve entry node from enumeration node
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                node_acquisition_mode_continuous):
            print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
            return False

        # Retrieve integer value from entry node
        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        self.camera.BeginAcquisition()

        res = []

        for i in range(frame_num):
            image_result = self.camera.GetNextImage(1000)
            if image_result.IsIncomplete():
                print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
            else:
                # width = image_result.GetWidth()
                # height = image_result.GetHeight()
                # image_converted = processor.Convert(image_result, PySpin.PixelFormat_Mono8)
                # -- Save -- #
                img_u8 = image_result.GetNDArray().copy()
                res.append(img_u8.astype(np.float32) / 255.0)
                image_result.Release()

        self.camera.EndAcquisition()
        return res

    def capture(self):
        res = self.capture_images(self.avg_frame)
        img_all = np.stack(res, axis=0)
        img = img_all.sum(axis=0) / self.avg_frame
        return (img * 255.0).astype(np.uint8)


class BaseProjector:
    def __init__(self):
        self.pat_size = None
        self.patterns = []
        self.current_idx = -1
        pass

    def get_pattern_num(self):
        return len(self.patterns)

    def set_pattern_set(self, pattern_set):
        self.pat_size = pattern_set[0].shape
        for pat in pattern_set:
            self.patterns.append(pat)
        return len(self.patterns)

    def project(self, idx):
        self.current_idx = idx
        return self.patterns[idx]


class VirtualProjector(BaseProjector):
    def __init__(self, rad, sigma):
        super().__init__()
        self.win = 2 * rad + 1
        self.sigma = sigma

    def set_pattern_set(self, pattern_set):
        pat_num = super().set_pattern_set(pattern_set)
        # Add gaussian filter to pattern.
        for i in range(pat_num):
            pat = plb.t2a(self.patterns[i])
            dst = cv2.GaussianBlur(pat, (self.win, self.win), self.sigma)
            self.patterns[i] = plb.a2t(dst)
        return pat_num


class ExtendProjector(BaseProjector):
    def __init__(self, screen_width=1920):
        super().__init__()
        self.win_name = 'Projected_Pattern'
        cv2.namedWindow(self.win_name)
        cv2.moveWindow(self.win_name, screen_width, 0)
        cv2.setWindowProperty(self.win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def project(self, idx):
        cv2.imshow(self.win_name, self.patterns[idx])
        cv2.waitKey(50)
        return super().project(idx)
