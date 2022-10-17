# -*- coding: utf-8 -*-

# @Time:      2022/10/6 11:35
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      intensity_visualization.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import pointerlib as plb


# - Coding Part - #
class IntensityVisualizer:

    def __init__(self, base_image, img_folders, pat_idx_set, color_set=None):
        self.base_image = base_image
        self.pat_idx_set = pat_idx_set
        self.img_size = (self.base_image.shape[1], self.base_image.shape[0])
        self.selected_coord = tuple([x // 2 for x in self.img_size])
        self.base_win_name = 'Image'

        self.img_sets = {}
        for tag, img_folder in img_folders.items():
            tmp = []
            for pat_idx in pat_idx_set:
                img = plb.imload(img_folder / f'img_{pat_idx}.png', flag_tensor=False)
                tmp.append((img * 255).astype(np.uint8))
            self.img_sets[tag] = np.stack(tmp, axis=0)

        self.color_set = color_set
        if color_set is None:
            self.color_set = {x: None for x in img_folders.keys()}

        pass

    def run(self):
        cv2.namedWindow(self.base_win_name)
        cv2.setMouseCallback(self.base_win_name, self.mouse_callback)
        plt.ion()
        plt.figure(figsize=(6, 8))
        plt.xlabel('Pattern Idx')
        plt.ylabel('Intensity')

        while True:
            # Draw red circle
            show_image = self.base_image.copy()
            cv2.circle(show_image, self.selected_coord, 5, (128, 128, 255), 2)

            # Draw plots
            plt.clf()
            w, h = self.selected_coord
            plt.title(f'h = {h}, w = {w}')
            for tag in self.img_sets.keys():
                intensity = self.img_sets[tag][:, h, w]
                color = self.color_set[tag]
                plt.plot(self.pat_idx_set, intensity, label=tag, color=color)
            # plt.legend()
            # plt.draw()
            plt.pause(0.2)

            cv2.imshow(self.base_win_name, show_image)
            key = cv2.waitKey(30)
            if key == '27':  # esc
                break

        cv2.destroyWindow(self.base_win_name)
        pass

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_coord = (x, y)
        pass


def visualize_texture():
    base_image = plb.imload(
        Path('C:/SLDataSet/20220907real/img/img_7.png'),
        flag_tensor=False
    )
    base_image = cv2.cvtColor((base_image * 255.0).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    img_folders = {
        'Pat Only': Path('C:/SLDataSet/20221005realsyn2/img'),
        'Syn': Path('C:/SLDataSet/20221005realsyn/img'),
        'Real': Path('C:/SLDataSet/20220907real/img'),
    }
    pat_idx_set = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)

    vis = IntensityVisualizer(base_image, img_folders, pat_idx_set)
    vis.run()


def visualize_epochs(data_folder, out_folder):
    out_folder_set = sorted(plb.subfolders(out_folder))

    base_image = plb.imload(
        out_folder_set[-1] / 'depth_viz.png',
        flag_tensor=False
    )
    base_image = (base_image * 255.0).astype(np.uint8)

    img_folders = {
        'Observed': data_folder,
    }
    color_set = {
        'Observed': None,
    }
    for i, out_folder_item in enumerate(out_folder_set):
        tag = out_folder_item.name
        img_folders[tag] = out_folder_item / 'img'
        alpha = (1 - i / len(out_folder_set)) * 0.95 + 0.05
        color_set[tag] = (alpha, alpha, alpha)

    pat_idx_set = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int32)

    vis = IntensityVisualizer(base_image, img_folders, pat_idx_set, color_set)
    vis.run()


def main():
    # visualize_texture()
    data_folder = Path('C:/SLDataSet/20220907real/img')
    out_folder = Path('C:/SLDataSet/20220907real-out/xyz2density-alpha-change-1000/output')
    visualize_epochs(data_folder, out_folder)


if __name__ == '__main__':
    main()
