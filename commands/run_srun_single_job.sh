#!/bin/sh

# GPU ressources:
srun --unbuffered \
  --gres=gpu:1 \
  --cpus-per-gpu=64 \
  --mem=50GB \
  --account=core-rad \
  singularity exec \
    --nv \
    --no-home \
    -B /projects/core-rad/tobweber/ddpm,/projects/core-rad/data/ILSVRC2012_img_train \
    /projects/containers/ngc-pytorch:21.09-py3 \
    "$@"