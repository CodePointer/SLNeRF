CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_train_lnx.ini \
    --pat_set 0,1,2,3,4,5,6 \
    --run_tag pat7

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_train_lnx.ini \
    --pat_set 0,1,2,3,4,5 \
    --run_tag pat6

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_train_lnx.ini \
    --pat_set 0,1,2,3,4 \
    --run_tag pat5

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_train_lnx.ini \
    --pat_set 0,1,2,3 \
    --run_tag pat4

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_train_lnx.ini \
    --pat_set 0,1,2 \
    --run_tag pat3

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_train_lnx.ini \
    --pat_set 1,2,3,4,5 \
    --run_tag pat12345

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_train_lnx.ini \
    --pat_set 0,1,3,5,6 \
    --run_tag pat01356

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_train_lnx.ini \
    --pat_set 0,1,3,4,6 \
    --run_tag pat01346

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_train_lnx.ini \
    --pat_set 0,1,4,5,6 \
    --run_tag pat01456

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_train_lnx.ini \
    --pat_set 1,3,4,6 \
    --run_tag pat1346

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_train_lnx.ini \
    --pat_set 0,2,3,5 \
    --run_tag pat0235

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_train_lnx.ini \
    --pat_set 0,1,2,6 \
    --run_tag pat0126


# Not regular
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/xyz2density_train_lnx.ini \
#     --pat_set 1,2,3,4,5 \
#     --train_dir /media/qiao/Videos/SLDataSet/20220907real \
#     --out_dir /media/qiao/Videos/SLDataSet/20220907real-outm6 \
#     --run_tag pat12345

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/xyz2density_train_lnx.ini \
#     --pat_set 1,2,3,4,5 \
#     --train_dir /media/qiao/Videos/SLDataSet/20221102real/scene_02 \
#     --out_dir /media/qiao/Videos/SLDataSet/20221102real/scene_02_main-m6 \
#     --run_tag pat12345

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/xyz2density_train_lnx.ini \
#     --pat_set 1,3,4,6 \
#     --train_dir /media/qiao/Videos/SLDataSet/20221102real/scene_02 \
#     --out_dir /media/qiao/Videos/SLDataSet/20221102real/scene_02_main-m6 \
#     --run_tag pat1346

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/xyz2density_train_lnx.ini \
#     --pat_set 1,3,4,6,0 \
#     --train_dir /media/qiao/Videos/SLDataSet/20221102real/scene_02 \
#     --out_dir /media/qiao/Videos/SLDataSet/20221102real/scene_02_main-m6 \
#     --run_tag pat13460

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/xyz2density_train_lnx.ini \
#     --pat_set 1,3,4,6,0 \
#     --train_dir /media/qiao/Videos/SLDataSet/20221102real/scene_02 \
#     --out_dir /media/qiao/Videos/SLDataSet/20221102real/scene_02_main-m6 \
#     --run_tag pat13460

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/xyz2density_train_lnx.ini \
#     --pat_set 1,4,5,0,6 \
#     --train_dir /media/qiao/Videos/SLDataSet/20221102real/scene_03 \
#     --out_dir /media/qiao/Videos/SLDataSet/20221028real/scene_03_main-m6 \
#     --run_tag pat14506