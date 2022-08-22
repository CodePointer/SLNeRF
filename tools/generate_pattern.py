# -*- coding: utf-8 -*-

# @Time:      2022/6/23 14:06
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      generate_pattern.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
import numpy as np
import cv2

from pathlib import Path


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


def draw_gray(gray_num, hei, wid):
    step = wid // len(gray_num)
    pat = gray_num.repeat(step, axis=0).reshape(1, -1).repeat(hei, axis=0).astype(np.uint8) * 255
    return pat


def generate_phase(interval, hei, wid):
    # intensity_base = 127.50
    # intensity_max = 127.50
    #
    # shift = np.array([0.0, (2 / 3) * np.pi, (4 / 3) * np.pi], dtype=np.float32).reshape(1, 3)
    # theta = (np.arange(0, interval) / interval * (2 * np.pi)).reshape(-1, 1)
    # phase_set = intensity_base + intensity_max * np.cos(theta + shift)
    #
    # step = wid // interval
    # phase_set_part = phase_set.reshape(1, interval, 3)
    # pats = np.tile(phase_set_part, [hei, step, 1])

    intensity_base = 127.50
    intensity_max = 127.50

    theta = (np.arange(1, interval + 1) / interval * (2 * np.pi) - np.pi).reshape(-1, 1)  # [0, interval] -> [-pi, pi]
    phi = np.array([0.0, (1 / 2) * np.pi, np.pi, (3 / 2) * np.pi], dtype=np.float32).reshape(1, 4)
    phase_set = intensity_base + intensity_max * np.sin(theta + phi)

    step = wid // interval
    phase_set_part = phase_set.reshape(1, interval, 4)
    pats = np.tile(phase_set_part, [hei, step, 1])

    return pats.astype(np.uint8)


def decode_phase(pats, interval):
    # pats = pats.astype(np.float32)
    # ntr = np.sqrt(3) * (pats[:, :, 1] - pats[:, :, 2])
    # dtr = 2 * pats[:, :, 0] - (pats[:, :, 1] + pats[:, :, 2])
    # # tan_val = ntr / dtr
    # theta = - np.arctan2(ntr, dtr)
    # theta[theta < 0.0] += 2 * np.pi
    # pixel_val = theta / (2 * np.pi) * interval

    pats = pats.astype(np.float32)
    ntr = (pats[:, :, 0] - pats[:, :, 2])
    dtr = (pats[:, :, 1] - pats[:, :, 3])
    # tan_val = ntr / dtr
    theta = np.arctan2(ntr, dtr)
    pixel_val = (theta + np.pi) / (2 * np.pi) * interval
    return pixel_val


def main():
    target_folder = Path('C:/SLDataSet/20220617s/pat')

    digit_num = 8
    hei = 800
    wid = 1280

    # gray_code = generate_gray(digit_num)
    # for i in range(digit_num):
    #     pat = draw_gray(gray_code[:, i], hei, wid)
    #     # cv2.imshow('pat', pat)
    #     # cv2.waitKey(10)
    #     cv2.imwrite(str(target_folder / f'pat_{2 * i}.png'), pat)
    #     pat_inv = 255 - pat
    #     cv2.imwrite(str(target_folder / f'pat_{2 * i + 1}.png'), pat_inv)

    interval = 40
    start_idx = 16
    pats = generate_phase(interval, hei, wid)
    pixel_val = decode_phase(pats, interval)
    # pix_show = (pixel_val * 6).astype(np.uint8)
    # cv2.imshow('pixel', pix_show)
    # cv2.waitKey(0)

    for i in range(4):
        pat = pats[:, :, i]
        cv2.imshow('pat', pat)
        cv2.waitKey(0)
        cv2.imwrite(str(target_folder / f'pat_{i + start_idx}.png'), pat)

    # pass


if __name__ == '__main__':
    main()
