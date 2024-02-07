DATA_DIR="/media/qiao/Videos/SLDataSet/SLNeuS/8_Dataset0801"
SCENE_IDX_LIST=( 04 05 06 )
PAT_SET_LIST=(
    "arr20r0,arr20r1,arr20r2,arr10r0,arr10r1,arr10r2,arr5r0,arr5r1,arr5r2"
    # "arr20r0,arr20r1,arr20r2,arr20r3,arr20r4,arr20r5"
    # "arr20r0,arr20r0inv,arr20r1,arr20r1inv,arr20r2,arr20r2inv,arr20r3,arr20r3inv,arr20r4,arr20r4inv,arr20r5,arr20r5inv"
    # "gc0,gc1,gc2,gc3,gc4,gc5,gc6,gc7,gc8"
    # "gc0,gc0inv,gc1,gc1inv,gc2,gc2inv,gc3,gc3inv,gc4,gc4inv,gc5,gc5inv,gc6,gc6inv,gc7,gc7inv,gc8,gc8inv,gc9,gc9inv"
)
RUN_TAG_LIST=(
    "main-online9-step"
    # "arr012345-on"
    # "arr012345inv-on"
    # "gc012345678-on"
    # "gc0123456789inv-on"
)
MILESTONE_LIST=(
    "0-3,1000-4,1500-5,2000-6,2500-7,3000-8,3500-9"
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
        MILESTONE=${MILESTONE_LIST[i]}

        CUDA_VISIBLE_DEVICES=${1} python -m torch.distributed.launch --nproc_per_node=1 --master_port=2950${1} main.py \
            --config ./params/xyz2sdf_train.ini \
            --train_dir ${TRAIN_DIR} \
            --out_dir ${OUT_DIR} \
            --pat_set ${PAT_SET} \
            --save_stone 50 \
            --patnum_scheduler [Milestone]${MILESTONE} \
            --warp_scheduler [Milestone]0-0.0,1000-1.0 \
            --run_tag ${RUN_TAG}

    done
done
