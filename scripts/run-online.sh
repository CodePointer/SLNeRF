DATA_DIR="/media/qiao/Videos/SLDataSet/SLNeuS/7_Dataset0531"
TRAIN_DIR=${DATA_DIR}"/scene_00"
OUT_DIR=${DATA_DIR}"-out/scene_00"
PAT_SET_LIST=(
    "arr20r0,arr20r1,arr20r2,arr20r3,arr20r4,arr20r5"
    "arr20r0,arr20r0inv,arr20r1,arr20r1inv,arr20r2,arr20r2inv,arr20r3,arr20r3inv,arr20r4,arr20r4inv,arr20r5,arr20r5inv"
    "gc0,gc1,gc2,gc3,gc4,gc5,gc6,gc7,gc8,gc9"
    "gc0,gc0inv,gc1,gc1inv,gc2,gc2inv,gc3,gc3inv,gc4,gc4inv,gc5,gc5inv,gc6,gc6inv,gc7,gc7inv,gc8,gc8inv,gc9,gc9inv"
)
RUN_TAG_LIST=(
    "arr012345-on"
    "arr012345inv-on"
    "gc0123456789-on"
    "gc0123456789inv-on"
)
MILESTONE_LIST=(
    "500-1,1000-2,1500-3,2000-4,2500-5,3000-6"
    "500-2,1000-4,1500-6,2000-8,2500-10,3000-12"
    "500-3,1000-4,1500-5,2000-6,2500-7,3000-8,3500-10"
    "500-6,1000-8,1500-10,2000-12,2500-14,3000-16,3500-20"
)

echo "Exp num: "${#PAT_SET_LIST[*]}" on CUDA "${1}", scene_00"${2}

for (( i=0; i<${#PAT_SET_LIST[*]}; i++ ))
do

    PAT_SET=${PAT_SET_LIST[i]}
    RUN_TAG=${RUN_TAG_LIST[i]}
    MILESTONE=${MILESTONE_LIST[i]}

    CUDA_VISIBLE_DEVICES=${1} python -m torch.distributed.launch --nproc_per_node=1 --master_port=2950${1} main.py \
        --config ./params/xyz2sdf_train.ini \
        --train_dir ${TRAIN_DIR} \
        --out_dir ${OUT_DIR} \
        --iter_start 0 \
        --pat_set ${PAT_SET} \
        --patnum_scheduler [Milestone]${MILESTONE} \
        --warp_scheduler [Milestone]0-0.0,1000-1.0 \
        --run_tag ${RUN_TAG}

    # tar -cf ${OUT_DIR}"/xyz2sdf-"${RUN_TAG}.tar" ${OUT_DIR}"/xyz2sdf-"${RUN_TAG}"/output/*"

done