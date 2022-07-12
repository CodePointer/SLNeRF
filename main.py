# -*- coding: utf-8 -*-

# - Package Imports - #
from args import get_args, post_process

from exps.exp_xy2depth import ExpXy2DepthWorker
from exps.exp_xyz2sdf import ExpXyz2SdfWorker


# - Coding Part - #
def get_worker(args):
    if args.argset == 'xy2depth':
        return ExpXy2DepthWorker(args)
    elif args.argset == 'xyz2sdf':
        return ExpXyz2SdfWorker(args)
    else:
        raise AssertionError(f'Invalid argset {args.argset}')


def main():
    args = get_args()
    post_process(args)

    worker = get_worker(args)
    worker.init_all()
    worker.do()


if __name__ == '__main__':
    main()
