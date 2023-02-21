# -*- coding: utf-8 -*-

# @Time:      2022/12/15 20:15
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      draw_distance_map.py
# @Software:  PyCharm
# @Description:
#   将pattern转换为distance map.

# - Package Imports - #
from pathlib import Path
import numpy as np

import pointerlib as plb


# - Coding Part - #
def draw_distance(center_file, distance_file=None):
    if distance_file is None:
        distance_file = center_file.parent / f'{center_file.stem}_dist{center_file.suffix}'
    mask_center = plb.imload(center_file, flag_tensor=False)
    max_step = 20

    hei, wid = mask_center.shape
    inf_val = max_step * 1.5
    distance_value = np.ones_like(mask_center) * inf_val
    main_queue = []
    for h, w in zip(*np.where(mask_center == 1.0)):
        distance_value[h, w] = 0.0
        main_queue.append([[h, w], [h, w], 0])

    while len(main_queue) > 0:
        coord_pt, blob_pt, step = main_queue[0]
        main_queue.pop(0)
        if step >= max_step:
            continue

        # Search neighbors
        for dh, dw in [[-1, -1], [-1, 0], [-1, 1],
                       [0, -1], [0, 1],
                       [1, -1], [1, 0], [1, 1]]:
            hn, wn = dh + coord_pt[0], dw + coord_pt[1]
            if 0 <= hn < hei and 0 <= wn < wid and distance_value[hn, wn] == inf_val:
                distance_value[hn, wn] = np.sqrt((hn - blob_pt[0]) ** 2 + (wn - blob_pt[1]) ** 2)
                main_queue.append([[hn, wn], blob_pt, step + 1])

    plb.imviz(distance_value, 'distance_map', 0, normalize=[0, inf_val])
    plb.imsave(distance_file, distance_value, scale=1e3, img_type=np.uint16)

    print('Finished.')


if __name__ == '__main__':
    draw_distance(Path(r'C:\SLDataSet\TADE\5_RealDataCut\pat_center.png'))
    # draw_distance(Path(r'C:\SLDataSet\20221213real\mask_center.png'))
