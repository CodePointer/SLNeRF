# -*- coding: utf-8 -*-

# @Time:      2023/5/24 16:49
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      gconly_coder.py.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
import numpy as np
from.basic_coder import GrayCodeCoder, load_img


# - Coding Part - #
class GCOnlyCoder:
    def __init__(self, hei, wid, gc_digit, interpolation=False):
        self.hei = hei
        self.wid = wid
        self.digit = gc_digit
        self.interval = wid / 2 ** gc_digit
        self.gc_coder = GrayCodeCoder(gc_digit)
        self.interpolation = interpolation

    def encode(self):
        gray_pats = self.gc_coder.create_pats(self.hei, self.wid)
        return gray_pats

    def decode(self, gray_pats, gray_pats_inv=None, img_base=None, mask=None):
        bin_interval = self.gc_coder.decode_imgs(gray_pats, gray_pats_inv, img_base)[0]
        if self.interpolation:
            res = np.zeros_like(bin_interval).astype(np.float32)
            mask = load_img(mask)[0].astype(np.float32) / 255.0
            for h in range(self.hei):
                inter_set = {}
                for w in range(self.wid):
                    bin_val = bin_interval[h, w]
                    if mask[h, w] == 0:
                        continue
                    if bin_val not in inter_set:
                        inter_set[bin_val] = [w, w + 1]
                    else:
                        inter_set[bin_val][1] = w + 1
                if len(inter_set.keys()) > 1:
                    for bin_val in range(max(inter_set.keys()) - 1):
                        if bin_val in inter_set and bin_val + 1 in inter_set:
                            if inter_set[bin_val][1] > inter_set[bin_val + 1][0]:
                                inter_set[bin_val][1] = inter_set[bin_val + 1][0]
                for bin_val in inter_set:
                    w_src, w_dst = inter_set[bin_val]
                    coord_src = bin_val * self.interval
                    for w in range(w_src, w_dst):
                        res[h, w] = (w - w_src) / (w_dst - w_src) * self.interval + coord_src
        else:
            res = (bin_interval.astype(np.float32) + 0.5) * self.interval
        return res


if __name__ == '__main__':
    pass
