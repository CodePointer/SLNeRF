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


def main():
    target_folder = Path('C:/SLDataSet/20220617s/scene000/pat')

    digit_num = 8
    hei = 800
    wid = 1280

    gray_code = generate_gray(digit_num)
    for i in range(digit_num):
        pat = draw_gray(gray_code[:, i], hei, wid)
        cv2.imshow('pat', pat)
        cv2.waitKey(10)
        cv2.imwrite(str(target_folder / f'pat_{i}.png'), pat)

    # pass


if __name__ == '__main__':
    main()
