# -*- coding: utf-8 -*-

# @Time:      2022/9/6 21:34
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      collect_data.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports -
import cv2
import numpy as np
from pathlib import Path

import pointerlib as plb
from tools.sensors import Flea3Camera, ExtendProjector


# - Coding Part - #
def wait_fun_real(camera, projector):
    target_hei, target_wid = 480, 640
    total_pattern_num = projector.get_pattern_num()
    projector.project(19)
    while True:
        img = camera.capture()
        img_viz = cv2.resize(img, (target_wid, target_hei), interpolation=cv2.INTER_NEAREST)

        hei, wid = img.shape
        hei_s, wid_s = hei // 16, wid // 16
        hei_bias, wid_bias = (hei - hei_s) // 2, (wid - wid_s) // 2
        img_part = img[hei_bias:hei_bias+hei_s, wid_bias:wid_bias+wid_s]
        img_part_viz = cv2.resize(img_part, (target_wid, target_hei), interpolation=cv2.INTER_NEAREST)

        show_viz = np.concatenate([img_viz, img_part_viz], axis=1)

        key = plb.imviz(show_viz, 'camera', wait=10)

        if key == 32:
            return True
        elif key == 27:  # esc
            return False


def wait_fun_ord(camera, projector):
    return True


def confirm_fun(res):
    key = plb.imviz_loop(res, name='camera', interval=500)
    if key == 27:
        return False
    elif key == 13:
        return True


def collect_data(camera, projector, wait_fun, confirm_fun, save_folder):
    while True:
        # Wait for collect
        continue_flag = wait_fun(camera, projector)
        if not continue_flag:
            break

        # Begin collection
        res = []
        for pat_idx in range(projector.get_pattern_num()):
            projector.project(pat_idx)
            img = camera.capture()  # uint8
            res.append(img.copy())

        # Confirm
        accept_flag = confirm_fun(res)

        # Save
        if accept_flag:
            for img_idx in range(len(res)):
                plb.imsave(save_folder / f'img_{img_idx}.png', res[img_idx], scale=1.0, mkdir=True)
            break


def collect_real_data(folder, scene_num):
    projector = ExtendProjector(screen_width=1920)
    pat_folder = folder / 'pat'
    total_frm = len(list(pat_folder.glob('*.png')))
    pat_set = []
    for i in range(total_frm):
        pat = plb.imload(pat_folder / f'pat_{i}.png', flag_tensor=False)
        pat_set.append((pat * 255.0).astype(np.uint8))
    projector.set_pattern_set(pat_set)
    camera = Flea3Camera(cam_idx=0, avg_frame=4)

    for scene_idx in range(scene_num):
        print(f'Start collecting scene {scene_idx:02}...')
        collect_data(
            camera, projector, wait_fun=wait_fun_real,
            confirm_fun=confirm_fun, save_folder=folder / f'scene_{scene_idx:02}' / 'img'
        )
        print(f'\tFinished.')


def main():
    folder = Path(r'C:\SLDataSet\20220907real')
    collect_real_data(folder, scene_num=1)
    pass


if __name__ == '__main__':
    main()
