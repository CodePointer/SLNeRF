DATA_DIR="/media/qiao/Videos/SLDataSet/SLNeuS/8_Dataset0801"
SCENE_IDX_LIST=( 00 01 02 03 )
PAT_SET_LIST=(
    "gc0,gc1,gc2"
    "gc0,gc1,gc2,gc3"
    "gc0,gc1,gc2,gc3,gc4"
    "gc0,gc1,gc2,gc3,gc4,gc5"
    "gc0,gc1,gc2,gc3,gc4,gc5,gc6"
    "gc0,gc1,gc2,gc3,gc4,gc5,gc6,gc7"
    "gc0,gc1,gc2,gc3,gc4,gc5,gc6,gc7,gc8"
    "gc0,gc0inv,gc1,gc1inv,gc2,gc2inv"
    "gc0,gc0inv,gc1,gc1inv,gc2,gc2inv,gc3,gc3inv"
    "gc0,gc0inv,gc1,gc1inv,gc2,gc2inv,gc3,gc3inv,gc4,gc4inv"
    "gc0,gc0inv,gc1,gc1inv,gc2,gc2inv,gc3,gc3inv,gc4,gc4inv,gc5,gc5inv"
    "gc0,gc0inv,gc1,gc1inv,gc2,gc2inv,gc3,gc3inv,gc4,gc4inv,gc5,gc5inv,gc6,gc6inv"
    "gc0,gc0inv,gc1,gc1inv,gc2,gc2inv,gc3,gc3inv,gc4,gc4inv,gc5,gc5inv,gc6,gc6inv,gc7,gc7inv"
    "gc0,gc0inv,gc1,gc1inv,gc2,gc2inv,gc3,gc3inv,gc4,gc4inv,gc5,gc5inv,gc6,gc6inv,gc7,gc7inv,gc8,gc8inv"
)
RUN_TAG_LIST=(
    "gc012"
    "gc0123"
    "gc01234"
    "gc012345"
    "gc0123456"
    "gc01234567"
    "gc012345678"
    "gc012inv"
    "gc0123inv"
    "gc01234inv"
    "gc012345inv"
    "gc0123456inv"
    "gc01234567inv"
    "gc012345678inv"
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

