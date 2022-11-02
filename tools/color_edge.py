# -*- coding: utf-8 -*-

# @Time:      2022/10/17 20:57
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      color_edge.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import pointerlib as plb


# - Coding Part - #
def main():
    main_path = Path('C:/SLDataSet/20221102real/scene_00')
    pat_set = [
        0,
        # 1,
        # 2,
        3,
        # 4,
        5,
        6
    ]

    # Load image set
    img_set = [plb.imload(main_path / 'img' / f'img_{idx}.png') for idx in range(0, 7)]
    img_set = torch.cat(img_set, dim=0)

    # Load depth
    depth_gt = plb.imload(main_path / f'depth_map.png', scale=10.0)
    mask = plb.imload(main_path / 'mask' / 'mask_occ.png')
    depth_dx = torch.abs(depth_gt[:, :-1, 1:] - depth_gt[:, :-1, :-1])
    mask_dx = mask[:, :-1, 1:] * mask[:, :-1, :-1]
    depth_dx *= mask_dx
    depth_dy = torch.abs(depth_gt[:, 1:, :-1] - depth_gt[:, :-1, :-1])
    mask_dy = mask[:, 1:, :-1] * mask[:, :-1, :-1]
    depth_dy *= mask_dy

    # Compute img distance
    img_dx = torch.abs(img_set[:, :-1, 1:] - img_set[:, :-1, :-1])
    img_dy = torch.abs(img_set[:, 1:, :-1] - img_set[:, :-1, :-1])

    img_dx = img_dx.sum(dim=0, keepdim=True) * mask_dx
    img_dy = img_dy.sum(dim=0, keepdim=True) * mask_dy

    img_dx[img_dx < 0.4] = 0.0
    img_dy[img_dy < 0.4] = 0.0

    def normalize(matrix, min_val=0.0, max_val=1.0):
        matrix = torch.clamp(matrix, min_val, max_val)
        return (matrix - min_val) / (max_val - min_val)

    # plb.imsave('depth_dx.png', normalize(depth_dx, max_val=10.0))
    # plb.imsave('depth_dy.png', normalize(depth_dy, max_val=10.0))
    # plb.imsave('img_dx.png', img_dx)
    # plb.imsave('img_dy.png', img_dy)

    # plb.imviz(depth_dx, 'depth_dx', 10, normalize=[0.0, 10.0])
    # plb.imviz(img_dx, 'img_dx', 10)
    # plb.imviz(depth_dy, 'depth_dy', 10, normalize=[0.0, 10.0])
    # plb.imviz(img_dy, 'img_dy', 10)

    # plt.hist(img_dx[img_dx > 0].numpy())
    # plt.hist(img_dy[img_dy > 0].numpy())
    # plt.show()

    img_edge = img_dx + img_dy
    img_edge[img_edge > 0.4] = 1.0
    img_edge[img_edge < 1.0] = 0.0
    plb.imviz(img_edge, 'img_edge', 0)

    print('finished.')

    pass


def my_softmax(input_tensor, dim):
    exp_tensor = torch.exp(input_tensor)
    sum_value = exp_tensor.sum(dim=dim, keepdim=True)
    return exp_tensor / sum_value


def softmax_test():
    x = torch.rand(10)
    x /= x.sum()
    x_soft = my_softmax(x, dim=0)
    x_soft2 = x.softmax(dim=0)
    # x_soft = x.softmax(dim=0)
    # x_soft2 = x_soft.softmax(dim=0)

    plt.plot(np.arange(10), x.numpy(), label='raw')
    plt.plot(np.arange(10), x_soft.numpy(), label='my_softmax')
    plt.plot(np.arange(10), x_soft2.numpy(), label='torch.softmax')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # softmax_test()
    main()
