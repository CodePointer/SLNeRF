DATA_DIR="/media/qiao/Videos/SLDataSet/SLNeuS/7_Dataset0531"
TRAIN_DIR=${DATA_DIR}"/scene_00"
OUT_DIR=${DATA_DIR}"-out/scene_00"
PAT_SET_LIST=(
    "gc0"
    "gc0,gc1"
    "gc0,gc1,gc2"
    "gc0,gc1,gc2,gc3"
    "gc0,gc1,gc2,gc3,gc4"
    "gc0,gc1,gc2,gc3,gc4,gc5"
    "gc0,gc0inv"
    "gc0,gc0inv,gc1,gc1inv"
    "gc0,gc0inv,gc1,gc1inv,gc2,gc2inv"
    "gc0,gc0inv,gc1,gc1inv,gc2,gc2inv,gc3,gc3inv"
    "gc0,gc0inv,gc1,gc1inv,gc2,gc2inv,gc3,gc3inv,gc4,gc4inv"
    "gc0,gc0inv,gc1,gc1inv,gc2,gc2inv,gc3,gc3inv,gc4,gc4inv,gc5,gc5inv"
)
RUN_TAG_LIST=(
    "gc0"
    "gc01"
    "gc012"
    "gc0123"
    "gc01234"
    "gc012345"
    "gc0inv"
    "gc01inv"
    "gc012inv"
    "gc0123inv"
    "gc01234inv"
    "gc012345inv"
)

echo "Exp num: "${#PAT_SET_LIST[*]}" on CUDA "${1}", scene_00"${2}

for (( i=0; i<${#PAT_SET_LIST[*]}; i++ ))
do

    PAT_SET=${PAT_SET_LIST[i]}
    RUN_TAG=${RUN_TAG_LIST[i]}

    CUDA_VISIBLE_DEVICES=${1} python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 main.py \
        --config ./params/xyz2sdf_train.ini \
        --train_dir ${TRAIN_DIR} \
        --out_dir ${OUT_DIR} \
        --iter_start 0 \
        --pat_set ${PAT_SET} \
        --patnum_scheduler -1 \
        --warp_scheduler [Milestone]0-0.0,1000-1.0 \
        --run_tag ${RUN_TAG}

done