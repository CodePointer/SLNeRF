# -*- coding: utf-8 -*-

# @Time:      2022/7/28 16:01
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      test_sample_pdf.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
import torch
import numpy as np
import matplotlib.pyplot as plt

from networks.neus import sample_pdf


# - Coding Part - #
def main():
    near = 300.0
    far = 900.0
    ray_samples = 64

    z_vals = torch.linspace(near, far, ray_samples, dtype=torch.float32)
    z_vals = z_vals.reshape(1, 64)

    weights = torch.ones_like(z_vals)[:, :-1] * 0.1
    z_step = (far - near) / ray_samples
    z_mid = (z_vals + z_step * 0.5)[:, :-1]

    # Change weights
    weights[:, 10:20] = 2.0
    weights[:, 40:50] = 4.0

    sampled_z = sample_pdf(z_vals, weights, 16, det=True)

    plt.plot(z_mid[0].numpy(), weights[0].numpy())
    # plt.plot(z_vals.numpy(), np.zeros_like(z_vals), 'ok')
    plt.plot(sampled_z[0].numpy(), np.zeros_like(sampled_z[0]), '*r')

    # plt.legend(['weight', 'sampled_z'])
    plt.show()

    print(sampled_z)


if __name__ == '__main__':
    main()