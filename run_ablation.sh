# Please set run_tag & pattern num in the ini file.

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_train_lnx.ini \
    --epoch_start 0 \
    --epoch_end 25001 \
    --alpha_stone 0-1.0,5000-0.5 \
    --lambda_stone 0-0.0 \
    --run_tag ours-warp

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_train_lnx.ini \
    --epoch_start 0 \
    --epoch_end 25001 \
    --alpha_stone 0-1.0,5000-0.5 \
    --lambda_stone 0-0.0,5000-1.0 \
    --ablation_tag ours-sample \
    --run_tag ours-sample

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_train_lnx.ini \
    --model_dir /media/qiao/Videos/SLDataSet/SLNeRF/scene_02-out/xyz2density-ours-warp/model/
    --epoch_start 5000 \
    --epoch_end 25001 \
    --alpha_stone 0-1.0,5000-0.5 \
    --lambda_stone 0-0.0,5000-1.0 \
    --run_tag ours

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_train_lnx.ini \
    --epoch_start 0 \
    --epoch_end 25001 \
    --alpha_stone 0-1.0,5000-0.5 \
    --lambda_stone 0-0.0 \
    --ablation_tag ours-sample \
    --run_tag vanilla

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/xyz2density_train_lnx.ini \
#     --out_dir /media/qiao/Videos/SLDataSet/20221102real/scene_00_ours \
#     --model_dir /media/qiao/Videos/SLDataSet/20221102real/scene_00_ours-reg/xyz2density-pat5/model \
#     --epoch_start 5001 \
#     --epoch_end 25001 \
#     --alpha_stone 0-1.0 \
#     --reg_stone 0-0.0,5000-0.001 \
#     --reg_color_sigma 1.0