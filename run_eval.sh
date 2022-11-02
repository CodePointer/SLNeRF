CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_eval_lnx.ini \
    --pat_set 0,1,2,3,4,5,6 \
    --run_tag pat7

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_eval_lnx.ini \
    --pat_set 0,1,2,3,4,5 \
    --run_tag pat6

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_eval_lnx.ini \
    --pat_set 0,1,2,3,4 \
    --run_tag pat5

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_eval_lnx.ini \
    --pat_set 0,1,2,3 \
    --run_tag pat4

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_eval_lnx.ini \
    --pat_set 0,1,2 \
    --run_tag pat3

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_eval_lnx.ini \
    --pat_set 1,2,3,4,5 \
    --run_tag 12345

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_eval_lnx.ini \
    --pat_set 0,1,3,5,6 \
    --run_tag 01356

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_eval_lnx.ini \
    --pat_set 0,1,3,4,6 \
    --run_tag 01346

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_eval_lnx.ini \
    --pat_set 0,1,4,5,6 \
    --run_tag 01456

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_eval_lnx.ini \
    --pat_set 1,3,4,6 \
    --run_tag 1346

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_eval_lnx.ini \
    --pat_set 0,2,3,5 \
    --run_tag 0235

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_eval_lnx.ini \
    --pat_set 0,1,2,6 \
    --run_tag 0126
