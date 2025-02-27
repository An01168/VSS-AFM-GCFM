#!/bin/bash



python -u train.py \
  --gpus 0,1,2,3 \
  --arch_encoder resnet18dilated \
  --arch_decoder focal_deepsup  \
  --fc_dim 512 \
  --num_epoch 20 \
  --epoch_iters 3000 \
  --batch_size_per_gpu 2 \
  --lr_encoder 0.01 \
  --lr_decoder 0.01 \
  --start_epoch 1


