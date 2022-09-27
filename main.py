# -*- coding: utf-8 -*-

# - Package Imports - #
from args import get_args, post_process

from exps.exp_xy2depth import ExpXy2DepthWorker
from exps.exp_xyz2sdf import ExpXyz2SdfWorker
from exps.exp_xyz2density import ExpXyz2DensityWorker


# - Coding Part - #
def get_worker(args):
    worker_set = {
        'xy2depth': ExpXy2DepthWorker,
        'xyz2sdf': ExpXyz2SdfWorker,
        'xyz2density': ExpXyz2DensityWorker
    }
    return worker_set[args.argset](args)
    # if args.argset == 'xy2depth':
    #     return ExpXy2DepthWorker(args)
    # elif args.argset == 'xyz2sdf':
    #     return ExpXyz2SdfWorker(args)
    # elif args.argset == 'xyz2density':
    #     return ExpXyz2DensityWorker(args)
    # else:
    #     raise AssertionError(f'Invalid argset {args.argset}')


def main():
    args = get_args()
    post_process(args)

    # for start_epoch in range(15001, 32001, 1000):
    #     args.epoch_start = start_epoch
    worker = get_worker(args)
    worker.init_all()
    worker.do()


if __name__ == '__main__':
    main()
