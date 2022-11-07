# -*- coding: utf-8 -*-

# @Time:      2022/11/3 0:09
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      copy_file.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
import os
import shutil
from pathlib import Path

import pointerlib as plb


# - Coding Part - #
def main():
    res_folder = Path('/media/qiao/Videos/SLDataSet/20221028real/scene_03_main-m6')
    target_folder = res_folder.parent / f'{res_folder.name}-zip'
    target_folder.mkdir(parents=True, exist_ok=True)

    exp_folders = [x for x in res_folder.glob('xyz2density-*') if x.is_dir()]
    for exp_folder in exp_folders:
        _, run_tag = exp_folder.name.split('-')
        # Get output folder
        epoch_outputs = [x for x in (exp_folder / 'output').glob('e_*') if x.is_dir()]
        epoch_outputs = epoch_outputs.sort(key=lambda x: int(x.name.split('_')[-1]))
        latest_output = epoch_outputs[-1]
        # Copy
        for file_name in [f'{run_tag}.png', f'{run_tag}.asc', f'{run_tag}_viz.png']:
            shutil.copy(
                src=str(latest_output / file_name),
                dst=str(target_folder / file_name)
            )


if __name__ == '__main__':
    main()
