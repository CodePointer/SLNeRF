# -*- coding: utf-8 -*-

# @Time:      2022/6/28 18:04
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      test_tensorboard.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
from tensorboard.backend.event_processing import event_accumulator

from pathlib import Path


# - Coding Part - #
def main():
    log_dir = Path(r'C:\SLDataSet\20220617s-out\xy2depth-20220628_172218\log')

    ea = event_accumulator.EventAccumulator(
        str(log_dir)
    )

    ea.Reload()

    print(ea)


if __name__ == '__main__':
    main()
