DATA_DIR="/media/qiao/Videos/SLDataSet/SLNeuS/7_Dataset0531"
TRAIN_DIR=${DATA_DIR}"/scene_00"
OUT_DIR=${DATA_DIR}"-out/scene_00"
PAT_SET_LIST=(
    # "arr20r0"
    # "arr20r0,arr20r1"
    "arr20r0,arr20r1,arr20r2"
    "arr20r0,arr20r1,arr20r2,arr20r3"
    "arr20r0,arr20r1,arr20r2,arr20r3,arr20r4"
    "arr20r0,arr20r1,arr20r2,arr20r3,arr20r4,arr20r5"
    # "arr20r0,arr20r0inv"
    # "arr20r0,arr20r0inv,arr20r1,arr20r1inv"
    "arr20r0,arr20r0inv,arr20r1,arr20r1inv,arr20r2,arr20r2inv"
    "arr20r0,arr20r0inv,arr20r1,arr20r1inv,arr20r2,arr20r2inv,arr20r3,arr20r3inv"
    "arr20r0,arr20r0inv,arr20r1,arr20r1inv,arr20r2,arr20r2inv,arr20r3,arr20r3inv,arr20r4,arr20r4inv"
    "arr20r0,arr20r0inv,arr20r1,arr20r1inv,arr20r2,arr20r2inv,arr20r3,arr20r3inv,arr20r4,arr20r4inv,arr20r5,arr20r5inv"
)
RUN_TAG_LIST=(
    # "arr0"
    # "arr01"
    # "arr012"
    "arr0123"
    "arr01234"
    "arr012345"
    # "arr0inv"
    # "arr01inv"
    "arr012inv"
    "arr0123inv"
    "arr01234inv"
    "arr012345inv"
)

echo "Exp num: "${#PAT_SET_LIST[*]}" on CUDA "${1}", scene_00"${2}

for (( i=0; i<${#PAT_SET_LIST[*]}; i++ ))
do

    PAT_SET=${PAT_SET_LIST[i]}
    RUN_TAG=${RUN_TAG_LIST[i]}

    CUDA_VISIBLE_DEVICES=${1} python -m torch.distributed.launch --nproc_per_node=1 --master_port=2950${1} main.py \
        --config ./params/xyz2sdf_train.ini \
        --train_dir ${TRAIN_DIR} \
        --out_dir ${OUT_DIR} \
        --iter_start 0 \
        --pat_set ${PAT_SET} \
        --patnum_scheduler -1 \
        --warp_scheduler [Milestone]0-0.0,1000-1.0 \
        --run_tag ${RUN_TAG}

done