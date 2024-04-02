DATA_DIR="/media/qiao/Videos/SLDataSet/SLNeuS/8_Dataset0801"
SCENE_IDX_LIST=( 04 05 06 )
PAT_SET_LIST=(
    # "arr20r0,arr20r1,arr20r2"
    # "arr40r0,arr20r0,arr10r0,arr5r0"
    # "arr20r0,arr20r0inv,arr10r0,arr10r1,arr5r0,arr5r1"
    "arr20r0,arr20r1,arr10r0,arr10r1,arr5r0,arr5r1"
    # "arr20r0,arr10r0,arr5r0,arr5r0inv"
    # "arr20r0,arr20r0inv,arr10r0,arr10r0inv,arr5r0,arr5r0inv"
    # "arr20r0,arr20r0inv,arr10r0,arr10r0inv,arr5r0,arr5r0inv,arr3r0,arr3r0inv"
    # "arr20r0,arr20r1,arr20r2,arr10r0,arr10r1,arr10r2,arr5r0,arr5r1,arr5r2"
    # "arr20r0,arr10r0,arr10r1,arr10r2,arr5r0,arr5r1,arr5r2"
    # "arr20r0,arr10r0,arr10r1,arr5r0,arr5r1,arr5r2"
    # "arr20r0,arr10r0,arr10r1,arr5r0,arr5r1"
)
RUN_TAG_LIST=(
    # "main_20x3"
    # "main_40_20_10_5"
    "main_20x2_10x2_5x2"
    # "dyn_20x2_10x2_5x2"
    # "main_20_10_5i"
    # "main_20i_10i_5i"
    # "main_20i_10i_5i_3i"
    # "dyn_20x3_10x3_5x3"
    # "main_20_10x3_5x3"
    # "main_20_10x2_5x3"
    # "main_20_10x2_5x2"
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
            --config ./params/xyz2sdf_train.ini \
            --train_dir ${TRAIN_DIR} \
            --out_dir ${OUT_DIR} \
            --pat_set ${PAT_SET} \
            --reflect_set "" \
            --run_tag ${RUN_TAG}

    done
done
