# multi = 12
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/xyz2density_train_lnx.ini \
#     --out_dir /media/qiao/Videos/SLDataSet/20220907real-out11/ours-reg \
#     --epoch_start 0 \
#     --epoch_end 25001 \
#     --alpha_stone 0-1.0 \
#     --reg_stone 0-0.0 \
#     --ablation_tag ours-reg \
#     --multires 12 \
#     --run_tag pat5m12
# multi = 6
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/xyz2density_train_lnx.ini \
#     --out_dir /media/qiao/Videos/SLDataSet/20220907real-out11/ours-reg \
#     --epoch_start 0 \
#     --epoch_end 25001 \
#     --alpha_stone 0-1.0 \
#     --reg_stone 0-0.0 \
#     --ablation_tag ours-reg \
#     --multires 6 \
#     --run_tag pat5m6
# # multi = 8
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/xyz2density_train_lnx.ini \
#     --out_dir /media/qiao/Videos/SLDataSet/20221102real/scene_01_ours-reg \
#     --epoch_start 0 \
#     --epoch_end 25001 \
#     --alpha_stone 0-1.0 \
#     --reg_stone 0-0.0 \
#     --ablation_tag ours-reg \
#     --multires 8 \
#     --run_tag pat5m8
# multi = 10
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_train_lnx.ini \
    --out_dir /media/qiao/Videos/SLDataSet/20220907real-out11/ours-reg \
    --epoch_start 0 \
    --epoch_end 25001 \
    --alpha_stone 0-1.0 \
    --reg_stone 0-0.0 \
    --ablation_tag ours-reg \
    --multires 10 \
    --run_tag pat5m10


# # multi = 12
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/xyz2density_train_lnx.ini \
#     --out_dir /media/qiao/Videos/SLDataSet/20221102real/scene_01_ours-color \
#     --model_dir /media/qiao/Videos/SLDataSet/20221102real/scene_01_ours-reg/xyz2density-pat5m12/model \
#     --epoch_start 5001 \
#     --alpha_stone 0-1.0 \
#     --reg_stone 0-0.0,5000-0.0001 \
#     --reg_color_sigma 1.0 \
#     --ablation_tag ours-color \
#     --multires 12 \
#     --run_tag pat5m12
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/xyz2density_train_lnx.ini \
#     --out_dir /media/qiao/Videos/SLDataSet/20221102real/scene_01_ours \
#     --model_dir /media/qiao/Videos/SLDataSet/20221102real/scene_01_ours-reg/xyz2density-pat5m12/model \
#     --epoch_start 5001 \
#     --alpha_stone 0-1.0 \
#     --reg_stone 0-0.0,5000-0.0001 \
#     --reg_color_sigma 1.0 \
#     --multires 12 \
#     --run_tag pat5m12

# multi = 6
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/xyz2density_train_lnx.ini \
#     --out_dir /media/qiao/Videos/SLDataSet/20220907real-out11/ours-color \
#     --model_dir /media/qiao/Videos/SLDataSet/20220907real-out11/ours-reg/xyz2density-pat5m6/model \
#     --epoch_start 5001 \
#     --alpha_stone 0-1.0 \
#     --reg_stone 0-0.0,5000-0.0001 \
#     --reg_color_sigma 1.0 \
#     --ablation_tag ours-color \
#     --multires 6 \
#     --run_tag pat5m6
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/xyz2density_train_lnx.ini \
#     --out_dir /media/qiao/Videos/SLDataSet/20220907real-out11/ours \
#     --model_dir /media/qiao/Videos/SLDataSet/20220907real-out11/ours-reg/xyz2density-pat5m6/model \
#     --epoch_start 5001 \
#     --alpha_stone 0-1.0 \
#     --reg_stone 0-0.0,5000-0.0001 \
#     --reg_color_sigma 1.0 \
#     --multires 6 \
#     --run_tag pat5m6

# # multi = 8
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/xyz2density_train_lnx.ini \
#     --out_dir /media/qiao/Videos/SLDataSet/20221102real/scene_01_ours-color \
#     --model_dir /media/qiao/Videos/SLDataSet/20221102real/scene_01_ours-reg/xyz2density-pat5m8/model \
#     --epoch_start 5001 \
#     --alpha_stone 0-1.0 \
#     --reg_stone 0-0.0,5000-0.0001 \
#     --reg_color_sigma 1.0 \
#     --ablation_tag ours-color \
#     --multires 8 \
#     --run_tag pat5m8
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/xyz2density_train_lnx.ini \
#     --out_dir /media/qiao/Videos/SLDataSet/20221102real/scene_01_ours \
#     --model_dir /media/qiao/Videos/SLDataSet/20221102real/scene_01_ours-reg/xyz2density-pat5m8/model \
#     --epoch_start 5001 \
#     --alpha_stone 0-1.0 \
#     --reg_stone 0-0.0,5000-0.0001 \
#     --reg_color_sigma 1.0 \
#     --multires 8 \
#     --run_tag pat5m8

# multi = 10
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_train_lnx.ini \
    --out_dir /media/qiao/Videos/SLDataSet/20220907real-out11/ours-color \
    --model_dir /media/qiao/Videos/SLDataSet/20220907real-out11/ours-reg/xyz2density-pat5m10/model \
    --epoch_start 5001 \
    --alpha_stone 0-1.0 \
    --reg_stone 0-0.0,5000-0.0001 \
    --reg_color_sigma 1.0 \
    --ablation_tag ours-color \
    --multires 10 \
    --run_tag pat5m10
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --config ./params/xyz2density_train_lnx.ini \
    --out_dir /media/qiao/Videos/SLDataSet/20220907real-out11/ours \
    --model_dir /media/qiao/Videos/SLDataSet/20220907real-out11/ours-reg/xyz2density-pat5m10/model \
    --epoch_start 5001 \
    --alpha_stone 0-1.0 \
    --reg_stone 0-0.0,5000-0.0001 \
    --reg_color_sigma 1.0 \
    --multires 10 \
    --run_tag pat5m10
