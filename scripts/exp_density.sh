TRAIN_DIR="/media/qiao/Videos/SLDataSet/SLNeuS/1_Collected0218"
OUT_DIR=${DATA_DIR}"-out"

#for (( i=0; i<=11; i++ ))
#do
     CUDA_VISIBLE_DEVICES=${1} python -m torch.distributed.launch --nproc_per_node=1 main.py \
         --config ./params/xyz2density_train.ini \
         --train_dir "${TRAIN_DIR}" \
         --out_dir "${OUT_DIR}" \
         --pat_set gc0,gc1,gc2,gc3,gc4,gc5 \
         --run_tag gc012345

    CUDA_VISIBLE_DEVICES=${1} python -m torch.distributed.launch --nproc_per_node=1 main.py \
        --config ./params/xyz2density_train.ini \
        --train_dir "${TRAIN_DIR}" \
         --out_dir "${OUT_DIR}" \
        --pat_set gc0,gc1,gc2,pm320n3i0,pm320n3i1,pm320n3i2 \
        --run_tag gc012pm320

    CUDA_VISIBLE_DEVICES=${1} python -m torch.distributed.launch --nproc_per_node=1 main.py \
        --config ./params/xyz2density_train.ini \
        --train_dir "${TRAIN_DIR}" \
         --out_dir "${OUT_DIR}" \
        --pat_set pm1280n3i0,pm1280n3i1,pm1280n3i2,pm80n3i0,pm80n3i1,pm80n3i2 \
        --run_tag pm1280pm80
#done
