DATA_DIR="/media/qiao/Videos/SLDataSet/SLNeuS/7_Dataset0531"
TRAIN_DIR=${DATA_DIR}"/scene_"${2}
OUT_DIR=${DATA_DIR}"-out/scene_"${2}
# PAT_SET_LIST=(
#     "rarray0"
#     "gc3,gc4,gc5"
# )
# RUN_TAG_LIST=(
#     "rarray0"
#     "gc3,gc4,gc5"
# )
# PAT_SET_LIST=(
#     "arr10r5,arr10r4,arr10r3,arr10r2,arr10r1,arr10r0"
#     "arr10r2,arr10r1,arr10r0"
#     "arr10r2,arr10r2inv,arr10r1,arr10r1inv,arr10r0,arr10r0inv"
#     "arr20r5,arr20r4,arr20r3,arr20r2,arr20r1,arr20r0"
#     "arr20r2,arr20r1,arr20r0"
# )
# RUN_TAG_LIST=(
#     "arr10r543210"
#     "arr10r210"
#     "arr10r210inv"
#     "arr20r543210"
#     "arr20r210"
# )
# PAT_SET_LIST=(
#     "arr10r5,arr10r5inv,arr10r4,arr10r4inv,arr10r3,arr10r3inv,arr10r2,arr10r2inv,arr10r1,arr10r1inv,arr10r0,arr10r0inv"
#     "arr20r5,arr20r5inv,arr20r4,arr20r4inv,arr20r3,arr20r3inv,arr20r2,arr20r2inv,arr20r1,arr20r1inv,arr20r0,arr20r0inv"
# )
# RUN_TAG_LIST=(
#     "arr10r543210inv"
#     "arr20r543210inv"
# )
PAT_SET_LIST=(
    "arr20r2,arr20r2inv,arr20r1,arr20r1inv,arr20r0,arr20r0inv"
)
RUN_TAG_LIST=(
    "arr20r210inv"
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
        --reflect_set "" \
        --pat_set ${PAT_SET} \
        --run_tag ${RUN_TAG}

    CUDA_VISIBLE_DEVICES=${1} python -m torch.distributed.launch --nproc_per_node=1 main.py \
        --config ./params/xyz2sdf_train.ini \
        --train_dir ${TRAIN_DIR} \
        --out_dir ${OUT_DIR} \
        --model_dir ${OUT_DIR}"/xyz2sdf-"${RUN_TAG}"/model" \
        --epoch_start 25001 \
        --reflect_set "" \
        --pat_set ${PAT_SET} \
        --run_tag ${RUN_TAG}"wrp"

done