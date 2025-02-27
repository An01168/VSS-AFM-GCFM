#!/bin/bash

#$ID_PATH = baseline-resnet18dilated-c1_deepsup-ngpus3-batchSize6-imgMaxSize1000-paddingConst8-segmDownsampleRate8-LR_encoder0.02-LR_decoder0.02-epoch20

# Inference
python -u eval_multipro.py \
  --gpus 0,1 \
  --id Ours-PSPNet50 \
  --suffix _epoch_20.pth \
  --arch_encoder resnet50dilated \
  --arch_decoder focal_deepsup \
  --fc_dim 2048 \
  --visualize
