DATA_DIR="/media/qiao/Videos/SLDataSet/SLNeuS/8_Dataset0801"
SCENE_IDX_LIST=( 00 01 02 03 04 05 )
PAT_SET="arr20r0,arr10r0,arr5r0,arr3r0"

echo "CUDA ["${1}"]: Exp loss; Scene num "${#SCENE_IDX_LIST[*]}

for SCENE_IDX in ${SCENE_IDX_LIST[*]}
do
    TRAIN_DIR=${DATA_DIR}"/scene_"${SCENE_IDX}
    OUT_DIR=${DATA_DIR}"-out/scene_"${SCENE_IDX}

    CUDA_VISIBLE_DEVICES=${1} python -m torch.distributed.launch --nproc_per_node=1 --master_port=2950${1} main.py \
        --config ./params/xyz2sdf_train.ini \
        --train_dir ${TRAIN_DIR} \
        --out_dir ${OUT_DIR} \
        --pat_set ${PAT_SET} \
        --reflect_set "" \
        --render_scheduler 1.0 \
        --igr_scheduler 0.0 \
        --warp_scheduler 0.0 \
        --run_tag "loss-rc"
    
    CUDA_VISIBLE_DEVICES=${1} python -m torch.distributed.launch --nproc_per_node=1 --master_port=2950${1} main.py \
        --config ./params/xyz2sdf_train.ini \
        --train_dir ${TRAIN_DIR} \
        --out_dir ${OUT_DIR} \
        --pat_set ${PAT_SET} \
        --reflect_set "" \
        --render_scheduler 1.0 \
        --igr_scheduler 0.1 \
        --warp_scheduler 0.0 \
        --run_tag "loss-rc-ek"

    CUDA_VISIBLE_DEVICES=${1} python -m torch.distributed.launch --nproc_per_node=1 --master_port=2950${1} main.py \
        --config ./params/xyz2sdf_train.ini \
        --train_dir ${TRAIN_DIR} \
        --out_dir ${OUT_DIR} \
        --pat_set ${PAT_SET} \
        --reflect_set "" \
        --render_scheduler 0.0 \
        --igr_scheduler 0.0 \
        --warp_scheduler 1.0 \
        --run_tag "loss-sc"
    
    CUDA_VISIBLE_DEVICES=${1} python -m torch.distributed.launch --nproc_per_node=1 --master_port=2950${1} main.py \
        --config ./params/xyz2sdf_train.ini \
        --train_dir ${TRAIN_DIR} \
        --out_dir ${OUT_DIR} \
        --pat_set ${PAT_SET} \
        --reflect_set "" \
        --render_scheduler 0.0 \
        --igr_scheduler 0.1 \
        --warp_scheduler 1.0 \
        --run_tag "loss-sc-ek"

    CUDA_VISIBLE_DEVICES=${1} python -m torch.distributed.launch --nproc_per_node=1 --master_port=2950${1} main.py \
        --config ./params/xyz2sdf_train.ini \
        --train_dir ${TRAIN_DIR} \
        --out_dir ${OUT_DIR} \
        --pat_set ${PAT_SET} \
        --reflect_set "" \
        --run_tag "loss-rc-sc-ek"
    
    done
done
