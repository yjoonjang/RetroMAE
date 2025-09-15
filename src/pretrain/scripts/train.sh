CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node 1 -m pretrain.run \
  --output_dir ../../MODELS/ \
  --data_dir /mnt/raid6/yjoonjang/projects/RetroMAE/examples/pretrain/pretrain_data/kure \
  --do_train True \
  --model_name_or_path skt/A.X-Encoder-base \
  --pretrain_method retromae