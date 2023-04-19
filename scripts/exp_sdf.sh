DATA_DIR="/media/qiao/Videos/SLDataSet/SLNeuS/1_Collected0218"
TRAIN_DIR=${DATA_DIR}"/scene_"${2}
OUT_DIR=${DATA_DIR}"-out/scene_"${2}
PAT_SET_LIST=(
    "gc0,gc1,gc2,gc3,gc4,gc5"
    "gc0,gc1,gc2,gc3,gc4"
    "gc0,gc1,gc2,gc3"
    "gc0,gc1,gc2"
)
RUN_TAG_LIST=(
    "gc012345"
    "gc01234"
    "gc0123"
    "gc012"
)

echo "Exp num: "${#PAT_SET_LIST[*]}" on CUDA "${1}", scene_"${2}

for (( i=0; i<${#PAT_SET_LIST[*]}; i++ ))
do

    PAT_SET=${PAT_SET_LIST[i]}
    RUN_TAG=${RUN_TAG_LIST[i]}

    CUDA_VISIBLE_DEVICES=${1} python -m torch.distributed.launch --nproc_per_node=1 main.py \
        --config ./params/xyz2sdf_train.ini \
        --train_dir ${TRAIN_DIR} \
        --out_dir ${OUT_DIR} \
        --lambda_stone 0-0.0 \
        --epoch_start 0 \
        --pat_set ${PAT_SET} \
        --run_tag ${RUN_TAG}

    CUDA_VISIBLE_DEVICES=${1} python -m torch.distributed.launch --nproc_per_node=1 main.py \
        --config ./params/xyz2sdf_train.ini \
        --train_dir ${TRAIN_DIR} \
        --out_dir ${OUT_DIR} \
        --model_dir ${OUT_DIR}"/xyz2sdf-"${RUN_TAG}"/model" \
        --epoch_start 25001 \
        --pat_set ${PAT_SET} \
        --run_tag ${RUN_TAG}"wrp"

done