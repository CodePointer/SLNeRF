DATA_DIR="/media/qiao/Videos/SLDataSet/SLNeuS/5_Dataset0509"
TRAIN_DIR=${DATA_DIR}"/scene_"${2}
OUT_DIR=${DATA_DIR}"-out/scene_"${2}
# PAT_SET_LIST=(
#     "rarray0"
#     "gc3,gc4,gc5"
# )
# RUN_TAG_LIST=(
#     "rarray0"
#     "gc3,gc4,gc5"
# )
PAT_SET_LIST=(
    "rarray0,rarray1,rarray2,rarray3,rarray4,rarray5"
    "rarray0,rarray1,rarray2"
)
RUN_TAG_LIST=(
    "rarray012345"
    "rarray012"
)
# PAT_SET_LIST=(
#     "gc3,gc4,gc5"
#     "rdot0,rdot1,rdot2,rdot3,rdot4,rdot5"
#     "rdot0,rdot1,rdot2"
# )
# RUN_TAG_LIST=(
#     "gc345"
#     "rdot012345"
#     "rdot012"
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
        --reflect_set "" \
        --run_tag ${RUN_TAG}"ref"

done