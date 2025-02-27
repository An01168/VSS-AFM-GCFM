#!/bin/bash



python -u train.py \
  --gpus 0,1,2,3 \
  --arch_encoder resnet50dilated \
  --arch_decoder focal_deepsup  \
  --fc_dim 2048 \
  --num_epoch 20 \
  --epoch_iters 6000 \
  --batch_size_per_gpu 1 \
  --lr_encoder 0.01 \
  --lr_decoder 0.01 \
  --start_epoch 1


