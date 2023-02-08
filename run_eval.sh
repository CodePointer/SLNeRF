echo "CUDA_VISIBLE_DEVICES: "${1}

for (( i = 60000; i <= 100000; i = i + 5000))
do
    CUDA_VISIBLE_DEVICES=${1} python -m torch.distributed.launch --nproc_per_node=1 main.py \
        --config ./params/xyz2sdf_eval.ini \
        --epoch_start $(expr $i + 1) \
        --epoch_end $(expr $i + 2)
done
