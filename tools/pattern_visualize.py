# -*- coding: utf-8 -*-

# @Time:      2023/8/12 11:26
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      pattern_visualize.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
from pathlib import Path
import numpy as np

import pointerlib as plb


# - Coding Part - #
def draw_patterns(patterns, hei_num, wid_num, row_major=True, interval=80, color=0.9):
    if len(patterns) == 0:
        return

    hei, wid = patterns[0].shape[-2:]

    hei_all = hei * hei_num + interval * (hei_num + 1)
    wid_all = wid * wid_num + interval * (wid_num + 1)
    canvas = np.ones([hei_all, wid_all], dtype=patterns[0].dtype) * color

    start_pos = []
    if row_major:
        for h in range(hei_num):
            for w in range(wid_num):
                start_pos.append((h * (hei + interval) + interval,
                                  w * (wid + interval) + interval))
    else:
        for w in range(wid_num):
            for h in range(hei_num):
                start_pos.append((h * (hei + interval) + interval,
                                  w * (wid + interval) + interval))

    for i, pat in enumerate(patterns):
        if i >= len(start_pos):
            break
        h, w = start_pos[i]
        canvas[h:h + hei, w:w + wid] = pat

    return canvas


def main():
    pat_folder = Path(r'C:\SLDataSet\SLNeRF\8_Dataset0801\pat')
    out_folder = pat_folder.parent

    # Online
    pat_tag = ['arr20r0', 'arr20r1', 'arr20r2', 'arr10r0', 'arr10r1',
               'arr10r2', 'arr5r0', 'arr5r1', 'arr5r2']
    kwargs = dict(hei_num=3, wid_num=3, row_major=False)
    out_name = 'online33'

    patterns = [plb.imload(pat_folder / f'pat_{tag}.png', flag_tensor=False) for tag in pat_tag]
    canvas = draw_patterns(patterns, interval=80, color=0.9, **kwargs)
    plb.imsave(out_folder / f'{out_name}.png', canvas)

    pass


if __name__ == '__main__':
    main()
