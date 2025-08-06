MODEL_NAME="skt/A.X-Encoder-base"
SHORT_MODEL_NAME="A.X-Encoder-base"

export WANDB_PROJECT="RetroMAE"
export WANDB_NAME="test-${SHORT_MODEL_NAME}-250806"

# # RetroMAE
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 \
#   -m pretrain.run \
#   --output_dir /mnt/raid6/yjoonjang/projects/RetroMAE/MODELS \
#   --data_dir /mnt/raid6/yjoonjang/projects/RetroMAE/examples/pretrain/pretrain_data/kure \
#   --do_train True \
#   --model_name_or_path skt/A.X-Encoder-base \
#   --max_seq_length 8192 \
#   --pretrain_method retromae \
#   --num_train_epochs 1 \
#   --learning_rate 1e-4 \
#   --report_to wandb \
#   --bf16 True \
#   --per_device_train_batch_size 2 \
#   --save_strategy epoch \
#   --save_steps 0.1 \
#   --logging_steps 1 \
#   --seed 42 \
#   --overwrite_output_dir False \
#   --optim adamw_torch \
#   --dataloader_num_workers 16 \
#   --warmup_ratio 0.1

# DupMAE
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 \
  -m pretrain.run \
  --output_dir /mnt/raid6/yjoonjang/projects/RetroMAE/MODELS/skt_A.X-Encoder-base-dupmae \
  --data_dir /mnt/raid6/yjoonjang/projects/RetroMAE/examples/pretrain/pretrain_data/kure \
  --do_train True \
  --model_name_or_path skt/A.X-Encoder-base \
  --max_seq_length 8192 \
  --pretrain_method dupmae \
  --num_train_epochs 1 \
  --learning_rate 1e-4 \
  --report_to wandb \
  --bf16 True \
  --per_device_train_batch_size 2 \
  --save_strategy epoch \
  --save_steps 0.1 \
  --logging_steps 1 \
  --seed 42 \
  --overwrite_output_dir False \
  --optim adamw_torch \
  --dataloader_num_workers 16 \
  --warmup_ratio 0.1