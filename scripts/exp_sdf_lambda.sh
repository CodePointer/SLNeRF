DATA_DIR="/media/qiao/Videos/SLDataSet/SLNeuS/3_Dataset0408"
TRAIN_DIR=${DATA_DIR}"/scene_01"
OUT_DIR=${DATA_DIR}"-out/scene_01"

echo "Exp on CUDA "${1}
PAT_SET="rdot0"
RUN_TAG="rdot0"

for lambda in 0.5 0.1 0.01 0.001 ;
do

    CUDA_VISIBLE_DEVICES=${1} python -m torch.distributed.launch --nproc_per_node=1 main.py \
        --config ./params/xyz2sdf_train.ini \
        --train_dir ${TRAIN_DIR} \
        --out_dir ${OUT_DIR} \
        --model_dir ${OUT_DIR}"/xyz2sdf-"${RUN_TAG}"/model" \
        --lambda_stone 0-0.0,25000-${lambda} \
        --epoch_start 25001 \
        --pat_set ${PAT_SET} \
        --run_tag ${RUN_TAG}"_lmd"${lambda}

done