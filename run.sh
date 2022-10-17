CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_train_lnx.ini \
    --pat_set 0,1,2,3,4,5,6,40,41
