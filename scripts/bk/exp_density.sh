DATA_DIR="/media/qiao/Videos/SLDataSet/SLNeuS/1_Collected0218"
TRAIN_DIR=${DATA_DIR}"/scene_"${2}
OUT_DIR=${DATA_DIR}"-out/scene_"${2}
PAT_SET_LIST=(
    "gc1,gc2,gc3,gc4,gc5,gc6"
    "gc0,gc1,gc3,gc4,gc5"
    "gc2,gc3,gc4,gc5"
    "gc1,gc3,gc5"
)
RUN_TAG_LIST=(
    "gc123456"
    "gc01345"
    "gc2345"
    "gc135"
)

echo "Exp num: "${#PAT_SET_LIST[*]}" on CUDA "${1}

for (( i=0; i<${#PAT_SET_LIST[*]}; i++ ))
do

    PAT_SET=${PAT_SET_LIST[i]}
    RUN_TAG=${RUN_TAG_LIST[i]}

    CUDA_VISIBLE_DEVICES=${1} python -m torch.distributed.launch --nproc_per_node=1 main.py \
        --config ./params/xyz2density_train.ini \
        --train_dir ${TRAIN_DIR} \
        --out_dir ${OUT_DIR} \
        --epoch_start 0 \
        --pat_set ${PAT_SET} \
        --run_tag ${RUN_TAG}

done