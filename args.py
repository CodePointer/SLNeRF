# -*- coding: utf-8 -*-

# - Package Imports - #
import os
import configargparse
import configparser
import torch
from torch.distributed import init_process_group
from pathlib import Path
from datetime import datetime


# - Coding Part - #
def get_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        is_config_file=True,
                        default='',
                        help='Path to config file.')
    parser.add_argument('--local_rank',
                        help='Node rank for distributed training.',
                        type=int,
                        default=0)

    # Required by base worker
    parser.add_argument('--argset',
                        help='Decide the worker for experiment.',
                        type=str)
    parser.add_argument('--debug_mode',
                        help='Check program is correct or not.',
                        action='store_true')
    parser.add_argument('--train_dir',
                        help='Folder for training data',
                        default='',
                        type=str)
    parser.add_argument('--out_dir',
                        help='The output directory',
                        default='./output',
                        type=str)
    parser.add_argument('--run_tag',
                        help='For output directory naming. Use timestamp by default.',
                        default='',
                        type=str)
    parser.add_argument('--test_dir',
                        help='Folder for testing data',
                        default='',
                        type=str)
    parser.add_argument('--model_dir',
                        help='Folder for the pre-trained model',
                        default='',
                        type=str)
    parser.add_argument('--exp_type',
                        help='Training / testing / online testing.',
                        default='train',
                        choices=['train', 'eval'],
                        type=str)
    parser.add_argument('--batch_num',
                        help='How many points are selected for each training.',
                        default=512,
                        type=int)
    parser.add_argument('--num_workers',
                        help='Workers for loading data. Will be set to 0 if under win systems.',
                        default=4,
                        type=int)
    parser.add_argument('--epoch_start',
                        help='Start epoch for training. If > 0, pre-trained model is needed.',
                        default=0,
                        type=int)
    parser.add_argument('--epoch_end',
                        help='End epoch for training.',
                        default=8,
                        type=int)
    parser.add_argument('--remove_history',
                        help='Save old model.pt or not. Useful when your model is very large.',
                        action='store_true')
    parser.add_argument('--lr',
                        help='Learning rate for model.',
                        default=1e-4,
                        type=float)
    parser.add_argument('--lr_step',
                        help='Learning-rate decay step. 0 for no decay. '
                             'When negative, lr will increase for lr-searching.',
                        default=0,
                        type=int)
    parser.add_argument('--report_stone',
                        help='How many iters for real-time report to be printed.',
                        default=32,
                        type=int)
    parser.add_argument('--img_stone',
                        help='How many iters for img visualization.',
                        default=256,
                        type=int)
    parser.add_argument('--model_stone',
                        help='How many epochs for model saving.',
                        default=1,
                        type=int)
    parser.add_argument('--save_stone',
                        help='How many iters for res writing. 0 for no result saving.',
                        default=0,
                        type=int)

    # Parameters in training
    parser.add_argument('--pat_set',
                        help='Pattern number set for training.',
                        default='',
                        type=str)
    parser.add_argument('--reflect_set',
                        help='Index of reflection images.',
                        default='',
                        type=str)
    parser.add_argument('--alpha_stone',
                        help='alpha value & shrink epoch for sampling.',
                        default='0-1.0',
                        type=str)
    parser.add_argument('--patch_rad',
                        help='Sampled patch radiace for training. Patch side length = 2 * patch_rad + 1',
                        default=0,
                        type=int)
    parser.add_argument('--reg_stone',
                        help='reg value & shrink epoch',
                        default='0-0',
                        type=str)
    parser.add_argument('--reg_color_sigma',
                        help='Parameters for color kernel control.',
                        default=0.4,
                        type=float)
    parser.add_argument('--peak_stone',
                        help='peak value & shink epoch',
                        default='0-0',
                        type=str)

    args = parser.parse_args()
    return args


def post_process(args):

    #
    # 1. Set run_tag & res_dir; + args.res_dir_name
    #
    if args.run_tag == '':
        args.run_tag = '{0:%Y%m%d_%H%M%S}'.format(datetime.now())
    args.res_dir_name = f'{args.argset}-{args.run_tag}'

    #
    # 2. Set model dir if model_dir is '_RERUN'
    #
    if args.model_dir == '_OUT_DIR_RERUN':
        args.model_dir = args.out_dir + f'/{args.argset}-{args.run_tag}' + '/model'
    # TODO: 临时修改
    args.model_dir = args.model_dir + f'/{args.argset}-{args.run_tag}/model'
    
    if args.exp_type == 'eval':
        args.epoch_end = args.epoch_start + 1
        args.report_stone = 1
        args.save_stone = 1

    #
    # 2. Write ini file to out_dir
    #
    out_config = Path(args.out_dir) / args.res_dir_name / 'params.ini'
    out_config.parent.mkdir(parents=True, exist_ok=True)
    save_parser = configparser.ConfigParser()
    save_parser.read_dict({'DEFAULT': vars(args)})
    with open(str(out_config), 'w+', encoding='utf-8') as file:
        save_parser.write(file)

    #
    # 3. Fix for windows system: num_worker -> 0
    #
    if os.name == 'nt':
        args.num_workers = 0

    #
    # 4. For distributed when you have multiple GPU cards. Only works for linux.
    #
    torch.cuda.set_device(args.local_rank)
    if os.name == 'nt':
        args.data_parallel = False
    else:
        init_process_group('nccl', init_method='env://')
        args.data_parallel = True
    args.device = torch.device(f'cuda:{args.local_rank}')

    return args
