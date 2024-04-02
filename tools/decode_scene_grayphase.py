# -*- coding: utf-8 -*-

# @Time:      2022/9/6 17:34
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      decode_scene_grayphase.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
import numpy as np
from pathlib import Path
from configparser import ConfigParser
import cv2

import pointerlib as plb


# - Coding Part - #
def generate_gray(digit_num):
    format_str = '{' + f':0{digit_num}b' + '}'
    total_num = 2 ** digit_num
    gray_list = []
    for idx in range(total_num):
        binary_str = format_str.format(idx)
        binary_num = np.array([int(x) for x in binary_str], dtype=np.int32)
        gray_num = binary_num.copy()
        for j in range(len(binary_num) - 1, 0, -1):
            if binary_num[j] == binary_num[j - 1]:
                gray_num[j] = 0
            else:
                gray_num[j] = 1
        gray_list.append(gray_num)
    gray_code = np.stack(gray_list, axis=0)
    return gray_code


def decode_gray(gray_pat, gray_pat_inv):
    assert len(gray_pat) == len(gray_pat_inv)
    digit_num = len(gray_pat)

    # 1. Generate gray2bin
    gray_code = generate_gray(digit_num)
    base_binary = 2 ** np.arange(digit_num - 1, -1, -1).reshape(1, -1)
    bin2gray = np.sum(gray_code * base_binary, axis=1)  # [2 ** digit_num, ]
    gray2bin = np.zeros_like(bin2gray)
    for bin_idx in range(bin2gray.shape[0]):
        gray_idx = bin2gray[bin_idx]
        gray2bin[gray_idx] = bin_idx

    # 2. Get 0-1 set
    pat_gray = np.stack(gray_pat, axis=0)  # [digit_num, H, W]
    pat_gray_inv = np.stack(gray_pat_inv, axis=0)
    gray_value = (pat_gray - pat_gray_inv > 0).astype(np.int32)
    base_binary = base_binary.reshape(-1, 1, 1)
    gray_decode = np.sum(gray_value * base_binary, axis=0)  # [H, W]
    bin_interval = np.take(gray2bin, gray_decode)

    return bin_interval


def visualize_pat_decode(pat_folder, gc_set):
    digit_num = len(gc_set)
    pat_set = np.stack([plb.imload(pat_folder / f'pat_gc{x}.png', flag_tensor=False) for x in gc_set], axis=0)

    gray_code = generate_gray(digit_num)
    base_binary = 2 ** np.arange(digit_num - 1, -1, -1).reshape(1, -1)
    bin2gray = np.sum(gray_code * base_binary, axis=1)  # [2 ** digit_num, ]
    gray2bin = np.zeros_like(bin2gray)
    for bin_idx in range(bin2gray.shape[0]):
        gray_idx = bin2gray[bin_idx]
        gray2bin[gray_idx] = bin_idx

    gray_decode = np.sum(pat_set * base_binary.reshape(-1, 1, 1), axis=0).astype(np.int32)
    bin_val = np.take(gray2bin, gray_decode)

    interval = 256 / 2 ** digit_num
    bin_val_viz = (bin_val * interval).astype(np.uint8)
    bin_val_rgb = cv2.applyColorMap(bin_val_viz, cv2.COLORMAP_JET)

    file_name = f'{digit_num}-' + ''.join(map(str, gc_set))
    cv2.imwrite(str(pat_folder.parent / 'viz' / f'grey_gc{file_name}.png'), bin_val_viz)
    cv2.imwrite(str(pat_folder.parent / 'viz' / f'rgb_gc{file_name}.png'), bin_val_rgb)
    pass


def main():
    pat_folder = Path(r'C:\SLDataSet\SLNeRF\1_Collected0218\pat')
    # decode(folder)
    gc_sets = [
        [0, 1, 2],
        [0, 2, 5],
        [1, 3, 5],
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [0, 1, 2, 3, 4],
        [0, 1, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 4, 5, 6],
        [1, 2, 3, 4, 5, 6]
    ]

    for gc_set in gc_sets:
        visualize_pat_decode(pat_folder, gc_set)


if __name__ == '__main__':
    main()
