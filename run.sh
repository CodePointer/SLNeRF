echo "CUDA_VISIBLE_DEVICES: "${1}

CUDA_VISIBLE_DEVICES=${1} python -m torch.distributed.launch --nproc_per_node=1 main.py \
    --config ./params/xyz2sdf_train.ini
