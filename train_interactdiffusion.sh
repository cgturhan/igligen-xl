#!/bin/bash -l

# Model Name
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"

# GPU Settings
NUM_GPUS=8  #TODO: change 
PARALLEL_PORT=21019
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Determine whether to use multi_gpu based on NUM_GPUS
if [ $NUM_GPUS -eq 1 ]; then
  MULTI_GPU=""
else
  MULTI_GPU="--multi_gpu"
fi

# Training Setting
BATCH_SIZE_SINGLE_GPU=8  # 2x2
NUM_WORKERS=16
DATA_CONFIG_PATH="dataset/sam_full_boxtext2img.yaml"
EXP_NAME=interactdiffusion_sdxl
# 250000 steps, validation_step 500, checkpointing_steps 1000
# "stabilityai/stable-diffusion-xl-base-1.0"
# "madebyollin/sdxl-vae-fp16-fix"
# Run scripts
  # --config $DATA_CONFIG_PATH \
accelerate launch $MULTI_GPU --num_processes=$NUM_GPUS --mixed_precision="fp16" \
  --num_machines 1 --dynamo_backend=no \
  --main_process_port $PARALLEL_PORT train_interactdiffusion_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --gligen_unet_path logs/backup-gligen-sdxl-512/checkpoint-600000/unet \
  --pretrained_vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \
  --resolution=512 \
  --train_batch_size $BATCH_SIZE_SINGLE_GPU \
  --gradient_accumulation_steps=1 \
  --mixed_precision="fp16" \
  --max_train_steps=250000 \
  --learning_rate=5.e-05 \
  --adam_weight_decay 0.0 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=1000 \
  --output_dir="logs/${EXP_NAME}" \
  --report_to=wandb \
  --data_path /home/user/jiuntian/data/data/hico_det_clip_sdxl_512 \
  --dataloader_num_workers $NUM_WORKERS \
  --validation_steps 500 \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps 1000 \
  --checkpoints_total_limit 5 \
  --prob_use_caption 0.5 \
  --prob_use_boxes 0.9 \
  --no_caption_only \
  --resume_from_checkpoint latest \
  --wandb_resume_id 0dmst6fm
