# -*- coding: utf-8 -*-

# - Package Imports - #
from args import get_args, post_process

# from exps.exp_xy2depth import ExpXy2DepthWorker
from exps.exp_xyz2sdf import ExpXyz2SdfWorker
# from exps.exp_xyz2density import ExpXyz2DensityWorker
# from exps.exp_vanilla import ExpVanillaNeRF
from exps.exp_classic import ExpClassicWorker
# from exps.exp_xyz2sdf_oneshot import ExpXyz2SdfOneShotWorker


# - Coding Part - #
def get_worker(args):
    worker_set = {
        # 'xy2depth': ExpXy2DepthWorker,
        'xyz2sdf': ExpXyz2SdfWorker,
        # 'xyz2density': ExpXyz2DensityWorker,
        # 'xyz2sdfoneshot': ExpXyz2SdfOneShotWorker,
        # 'vanilla': ExpVanillaNeRF,
        'ClassicBFH': ExpClassicWorker,
        'ClassicBFN': ExpClassicWorker,
        'ClassicGCC': ExpClassicWorker,
        'ClassicGrayOnly': ExpClassicWorker,
    }
    assert args.argset in worker_set.keys()
    return worker_set[args.argset](args)


def main():
    args = get_args()
    post_process(args)

    # TODO: Check if exists
    worker = get_worker(args)
    worker.init_all()
    worker.do()


if __name__ == '__main__':
    main()
