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
# PAT_SET_LIST=(
#     "rdot0,rdot1,rdot2,rdot3,rdot4,rdot5"
#     "rdot0,rdot1,rdot2,rdot3,rdot4"
#     "rdot0,rdot1,rdot2,rdot3"
#     "rdot0,rdot1,rdot2"
#     "rdot0,rdot1"
# )
# RUN_TAG_LIST=(
#     "rdot012345"
#     "rdot01234"
#     "rdot0123"
#     "rdot012"
#     "rodt01"
# )
# PAT_SET_LIST=(
#     "rletter0,rletter1,rletter2,rletter3,rletter4,rletter5"
#     "rletter0,rletter1,rletter2,rletter3,rletter4"
#     "rletter0,rletter1,rletter2,rletter3"
#     "rletter0,rletter1,rletter2"
#     "rletter0,rletter1"
# )
# RUN_TAG_LIST=(
#     "rletter012345"
#     "rletter01234"
#     "rletter0123"
#     "rletter012"
#     "rletter01"
# )

echo "Exp num: "${#PAT_SET_LIST[*]}" on CUDA "${1}", scene_"${2}

for (( i=0; i<${#PAT_SET_LIST[*]}; i++ ))
do

    PAT_SET=${PAT_SET_LIST[i]}
    RUN_TAG=${RUN_TAG_LIST[i]}

    CUDA_VISIBLE_DEVICES=${1} python -m torch.distributed.launch --nproc_per_node=1 main.py \
        --config ./params/xyz2sdf_train.ini \
        --train_dir ${TRAIN_DIR} \
        --out_dir ${OUT_DIR} \
        --epoch_start 0 \
        --pat_set ${PAT_SET} \
        --run_tag ${RUN_TAG}

done