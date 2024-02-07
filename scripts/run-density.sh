DATA_DIR="/media/qiao/Videos/SLDataSet/SLNeuS/8_Dataset0801"
SCENE_IDX_LIST=( 01 02 03 04 05 06 )
PAT_SET_LIST=(
    "arr20r0,arr20r1,arr10r0,arr10r1,arr5r0,arr5r1"
)
RUN_TAG_LIST=(
    "main_20x2_10x2_5x2"
)

echo "CUDA ["${1}"]: Exp num "${#PAT_SET_LIST[*]}"; Scene num "${#SCENE_IDX_LIST[*]}

for SCENE_IDX in ${SCENE_IDX_LIST[*]}
do
    TRAIN_DIR=${DATA_DIR}"/scene_"${SCENE_IDX}
    OUT_DIR=${DATA_DIR}"-out/scene_"${SCENE_IDX}

    for (( i=0; i<${#PAT_SET_LIST[*]}; i++ ))
    do
        PAT_SET=${PAT_SET_LIST[i]}
        RUN_TAG=${RUN_TAG_LIST[i]}

        CUDA_VISIBLE_DEVICES=${1} python -m torch.distributed.launch --nproc_per_node=1 --master_port=2950${1} main.py \
            --config ./params/xyz2density_train.ini \
            --train_dir ${TRAIN_DIR} \
            --out_dir ${OUT_DIR} \
            --pat_set ${PAT_SET} \
            --reflect_set "" \
            --run_tag ${RUN_TAG}

    done
done
