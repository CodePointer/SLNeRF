# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/xyz2density_train_lnx.ini \
#     --pat_set 0,1,2,3,4,5,6 \
#     --run_tag pat7

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/xyz2density_train_lnx.ini \
#     --pat_set 0,1,2,3,4,5 \
#     --run_tag pat6

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/xyz2density_train_lnx.ini \
#     --pat_set 0,1,2,3,4 \
#     --run_tag pat5

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/xyz2density_train_lnx.ini \
#     --pat_set 0,1,2,3 \
#     --run_tag pat4

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/xyz2density_train_lnx.ini \
#     --pat_set 0,1,2 \
#     --run_tag pat3

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/xyz2density_train_lnx.ini \
#     --pat_set 0,1,2,3,4 \
#     --run_tag pat01234

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/xyz2density_train_lnx.ini \
#     --pat_set 1,2,3,4,5 \
#     --reg_stone 0-0.0,5000-0.001 \
#     --run_tag pat12345

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/xyz2density_train_lnx.ini \
#     --pat_set 1,2,3,4,6 \
#     --reg_stone 0-0.0,5000-0.001 \
#     --run_tag pat12346

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_train_lnx.ini \
    --pat_set 1,3,4,6 \
    --reg_stone 0-0.0,5000-0.001 \
    --run_tag pat1346