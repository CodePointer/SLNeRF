DATA_DIR="/media/qiao/Videos/SLDataSet/SLNeuS/3_Dataset0408"
TRAIN_DIR=${DATA_DIR}"/scene_"${2}
OUT_DIR=${DATA_DIR}"-out/scene_"${2}
PAT_SET_LIST=(
    "rdot0"
    "rletter0"
)
RUN_TAG_LIST=(
    "rdot0"
    "rletter0"
)

echo "Exp num: "${#PAT_SET_LIST[*]}" on CUDA "${1}", scene_"${2}

for (( i=0; i<${#PAT_SET_LIST[*]}; i++ ))
do

    PAT_SET=${PAT_SET_LIST[i]}
    RUN_TAG=${RUN_TAG_LIST[i]}

    CUDA_VISIBLE_DEVICES=${1} python -m torch.distributed.launch --nproc_per_node=1 main.py \
        --config ./params/xyz2sdfoneshot_train.ini \
        --train_dir ${TRAIN_DIR} \
        --out_dir ${OUT_DIR} \
        --epoch_start 0 \
        --pat_set ${PAT_SET} \
        --run_tag ${RUN_TAG}

done