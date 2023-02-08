# -*- coding: utf-8 -*-

# - Package Imports - #
from args import get_args, post_process

# from exps.exp_xy2depth import ExpXy2DepthWorker
from exps.exp_xyz2sdf import ExpXyz2SdfWorker
# from exps.exp_xyz2density import ExpXyz2DensityWorker
# from exps.exp_vanilla import ExpVanillaNeRF


# - Coding Part - #
def get_worker(args):
    worker_set = {
        # 'xy2depth': ExpXy2DepthWorker,
        'xyz2sdf': ExpXyz2SdfWorker,
        # 'xyz2density': ExpXyz2DensityWorker,
        # 'vanilla': ExpVanillaNeRF,
    }
    assert args.argset in worker_set.keys()
    return worker_set[args.argset](args)


def main():
    args = get_args()
    post_process(args)

    worker = get_worker(args)
    worker.init_all()
    worker.do()


if __name__ == '__main__':
    main()
